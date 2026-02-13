#pragma once

#include "mini_nccl.h"
#include "Socket.h"
#include <infiniband/verbs.h>
#include <vector>
#include <mutex>
#include <map>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <unordered_set>

namespace mini_nccl {

struct RdmaInfo {
    int rank;
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
};

class RDMAMemoryRegion : public MemoryRegion {
public:
    RDMAMemoryRegion(struct ibv_pd* pd, void* ptr, size_t size) : ptr_(ptr), size_(size) {
        mr_ = ibv_reg_mr(pd, ptr, size, 
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_) throw std::runtime_error("Failed to register MR");
    }
    ~RDMAMemoryRegion() { if (mr_) ibv_dereg_mr(mr_); }
    void* ptr() const override { return ptr_; }
    size_t size() const override { return size_; }
    uint32_t lkey() const { return mr_->lkey; }
private:
    void* ptr_;
    size_t size_;
    struct ibv_mr* mr_;
};

class RDMATransport;

// RDMA 请求对象：用于追踪 isend/irecv 的状态
class RDMARequest : public Request {
public:
    RDMARequest(RDMATransport* transport, uint64_t wr_id) 
        : transport_(transport), wr_id_(wr_id), completed_(false) {}

    // 等待直到完成
    void wait() override; 
    
    // 检查是否完成
    bool isCompleted() const override { return completed_; }
    
    // 内部使用：标记完成
    void markCompleted() { completed_ = true; }
    uint64_t id() const { return wr_id_; }

private:
    RDMATransport* transport_;
    uint64_t wr_id_;
    volatile bool completed_;
};

class RDMATransport : public Transport {
public:
    RDMATransport(int rank, int nRanks, std::string root_ip = "127.0.0.1") 
        : rank_(rank), nRanks_(nRanks), root_ip_(root_ip) {
        setup_device();
    }

    ~RDMATransport() {
        // 清理资源...
    }

    // 复用之前的 init 逻辑
    void init() override {
        create_qps();
        exchange_and_connect(); 
    }

    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) override {
        return std::make_shared<RDMAMemoryRegion>(pd_, ptr, size);
    }

    // =============================================================
    // 核心实现：异步发送 (Isend)
    // =============================================================
    std::shared_ptr<Request> isend(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) override {
        auto rmr = std::static_pointer_cast<RDMAMemoryRegion>(mr);
        uint64_t wr_id = next_wr_id_++;
        
        struct ibv_sge sge;
        sge.addr = (uint64_t)rmr->ptr() + offset;
        sge.length = length;
        sge.lkey = rmr->lkey();

        struct ibv_send_wr wr = {};
        wr.wr_id = wr_id;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;

        struct ibv_send_wr* bad_wr;
        if (ibv_post_send(qps_[rank], &wr, &bad_wr)) {
            throw std::runtime_error("ibv_post_send failed");
        }

        return std::make_shared<RDMARequest>(this, wr_id);
    }

    // =============================================================
    // 核心实现：异步接收 (Irecv)
    // =============================================================
    std::shared_ptr<Request> irecv(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) override {
        auto rmr = std::static_pointer_cast<RDMAMemoryRegion>(mr);
        uint64_t wr_id = next_wr_id_++;

        struct ibv_sge sge;
        sge.addr = (uint64_t)rmr->ptr() + offset;
        sge.length = length;
        sge.lkey = rmr->lkey();

        struct ibv_recv_wr wr = {};
        wr.wr_id = wr_id;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        struct ibv_recv_wr* bad_wr;
        if (ibv_post_recv(qps_[rank], &wr, &bad_wr)) {
            throw std::runtime_error("ibv_post_recv failed");
        }

        return std::make_shared<RDMARequest>(this, wr_id);
    }

    // =============================================================
    // 轮询引擎 (Polling Engine)
    // =============================================================
    // 这是一个关键函数：它去 CQ 里捞数据，捞到了就去更新对应的 Request 状态
    void poll() {
        struct ibv_wc wc[16];
        int n = ibv_poll_cq(cq_, 16, wc);
        if (n < 0) throw std::runtime_error("poll_cq failed");

        for (int i = 0; i < n; ++i) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                std::cerr << "WC Error: " << ibv_wc_status_str(wc[i].status) << std::endl;
                throw std::runtime_error("Work Completion Error");
            }
            // 将完成的 ID 加入集合
            completed_ids_.insert(wc[i].wr_id);
        }
    }

    // 检查某个 ID 是否完成
    bool check_completion(uint64_t wr_id) {
        poll(); // 每次检查前都尝试捞一把
        if (completed_ids_.count(wr_id)) {
            // 如果完成了，为了节省内存，可以删掉它（这就要求 Request 只能 wait 一次）
            // 简单起见，这里不删，或者用更复杂的数据结构管理
            return true;
        }
        return false;
    }

private:
    int rank_;
    int nRanks_;
    std::string root_ip_;
    struct ibv_context* ctx_ = nullptr;
    struct ibv_pd* pd_ = nullptr;
    struct ibv_cq* cq_ = nullptr;
    std::map<int, struct ibv_qp*> qps_;
    
    uint64_t next_wr_id_ = 0;
    std::unordered_set<uint64_t> completed_ids_; // 已完成的任务ID池

    void setup_device() {
        int num_devices;
        struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list) throw std::runtime_error("No RDMA devices");
        struct ibv_device* device = nullptr;
        for (int i = 0; i < num_devices; ++i) {
            if (std::string(ibv_get_device_name(dev_list[i])) == "rxe0") {
                device = dev_list[i]; break;
            }
        }
        if (!device) throw std::runtime_error("rxe0 not found");
        ctx_ = ibv_open_device(device);
        pd_ = ibv_alloc_pd(ctx_);
        cq_ = ibv_create_cq(ctx_, 1024, nullptr, nullptr, 0); // 加大 CQ 深度
        ibv_free_device_list(dev_list);
    }

    void create_qps() {
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) continue;
            struct ibv_qp_init_attr attr = {};
            attr.send_cq = cq_;
            attr.recv_cq = cq_;
            attr.qp_type = IBV_QPT_RC;
            attr.cap.max_send_wr = 1024; // 加大队列深度
            attr.cap.max_recv_wr = 1024;
            attr.cap.max_send_sge = 1;
            attr.cap.max_recv_sge = 1;
            qps_[i] = ibv_create_qp(pd_, &attr);
        }
    }

    void exchange_and_connect() {
        // 1. Query Port & GID
        struct ibv_port_attr port_attr;
        ibv_query_port(ctx_, 1, &port_attr);
        union ibv_gid my_gid;
        ibv_query_gid(ctx_, 1, 1, &my_gid);

        // 2. Prepare Infos
        std::vector<RdmaInfo> my_infos;
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) my_infos.push_back({rank_, 0, 0, {0}});
            else {
                my_infos.push_back({rank_, qps_[i]->qp_num, port_attr.lid, {0}});
                memcpy(my_infos.back().gid, my_gid.raw, 16);
            }
        }

        // 3. TCP Exchange
        std::vector<std::vector<RdmaInfo>> global_registry(nRanks_);
        if (rank_ == 0) {
            ServerSocket server(8888);
            std::vector<std::shared_ptr<Socket>> clients;
            global_registry[0] = my_infos;
            for (int i = 1; i < nRanks_; ++i) {
                auto sock = server.accept();
                clients.push_back(sock);
                int r; sock->recv(&r, sizeof(int));
                std::vector<RdmaInfo> peer_infos(nRanks_);
                sock->recv(peer_infos.data(), nRanks_ * sizeof(RdmaInfo));
                global_registry[r] = peer_infos;
            }
            for (auto& sock : clients) {
                for (int i = 0; i < nRanks_; ++i) 
                    sock->send(global_registry[i].data(), nRanks_ * sizeof(RdmaInfo));
            }
        } else {
            auto sock = connect_to(root_ip_, 8888);
            sock->send(&rank_, sizeof(int));
            sock->send(my_infos.data(), nRanks_ * sizeof(RdmaInfo));
            for(int i=0; i<nRanks_; ++i) {
                global_registry[i].resize(nRanks_);
                sock->recv(global_registry[i].data(), nRanks_ * sizeof(RdmaInfo));
            }
        }

        // 4. Connect
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) continue;
            connect_qp(qps_[i], global_registry[i][rank_]);
        }
        std::cout << "[RDMA] Bootstrap Done." << std::endl;
    }

    void connect_qp(struct ibv_qp* qp, RdmaInfo info) {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = 1;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024;
        attr.dest_qp_num = info.qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = info.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = 1;
        memcpy(attr.ah_attr.grh.dgid.raw, info.gid, 16);
        attr.ah_attr.grh.sgid_index = 1;
        attr.ah_attr.grh.hop_limit = 1;
        ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    }
};

// Request 的 wait 实现必须放在 Transport 定义之后
inline void RDMARequest::wait() {
    while (!completed_) {
        // 忙等待：不断让 transport 去 poll CQ
        if (transport_->check_completion(wr_id_)) {
            completed_ = true;
        }
    }
}

} // namespace mini_nccl