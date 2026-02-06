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

class RDMATransport; // Forward decl

// --- Request 实现 (支持复用) ---
class RDMARequest : public Request {
public:
    RDMARequest() : transport_(nullptr), wr_id_(0), completed_(false), pool_idx_(-1) {}

    // 重置状态 (替代构造函数)
    void reset(RDMATransport* transport, uint64_t wr_id, int pool_idx) {
        transport_ = transport;
        wr_id_ = wr_id;
        pool_idx_ = pool_idx;
        completed_ = false;
    }

    void wait() override; 
    void release() override; // 归还自己

    bool isCompleted() const override { return completed_; }
    
    // 供 Transport 更新状态
    void markCompleted() { completed_ = true; }
    uint64_t id() const { return wr_id_; }

private:
    RDMATransport* transport_;
    uint64_t wr_id_;
    volatile bool completed_;
    int pool_idx_; 
};

class RDMATransport : public Transport {
public:
    RDMATransport(int rank, int nRanks, std::string root_ip = "127.0.0.1") 
        : rank_(rank), nRanks_(nRanks), root_ip_(root_ip) {
        setup_device();
        
        // --- 内存池初始化 ---
        // 预分配 4096 个请求对象，足够跑满流水线
        // 使用 vector 保证内存连续性 (Cache Friendly)
        int pool_size = 4096; 
        request_pool_.resize(pool_size);
        free_indices_.reserve(pool_size);
        for (int i = 0; i < pool_size; ++i) {
            free_indices_.push_back(i);
        }
    }

    ~RDMATransport() {
        for (auto& pair : qps_) if (pair.second) ibv_destroy_qp(pair.second);
        if (cq_) ibv_destroy_cq(cq_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (ctx_) ibv_close_device(ctx_);
    }

    void init() override {
        create_qps();
        exchange_and_connect();
    }

    // --- 对象池分配 (O(1)) ---
    RDMARequest* allocateRequest(uint64_t wr_id) {
        if (free_indices_.empty()) {
            throw std::runtime_error("RDMA Request Pool Exhausted! Leak or pool too small.");
        }
        int idx = free_indices_.back();
        free_indices_.pop_back();

        RDMARequest* req = &request_pool_[idx];
        req->reset(this, wr_id, idx);
        return req;
    }

    // --- 对象池回收 (O(1)) ---
    void freeRequest(int idx) {
        free_indices_.push_back(idx);
    }

    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) override {
        return std::make_shared<RDMAMemoryRegion>(pd_, ptr, size);
    }

    // Isend (无锁，无 malloc)
    Request* isend(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) override {
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

        return allocateRequest(wr_id);
    }

    // Irecv (无锁，无 malloc)
    Request* irecv(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) override {
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

        return allocateRequest(wr_id);
    }

    void poll() {
        struct ibv_wc wc[16];
        int n = ibv_poll_cq(cq_, 16, wc);
        if (n < 0) throw std::runtime_error("poll_cq failed");

        for (int i = 0; i < n; ++i) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                std::cerr << "WC Error: " << ibv_wc_status_str(wc[i].status) << std::endl;
                throw std::runtime_error("Work Completion Error");
            }
            completed_ids_.insert(wc[i].wr_id);
        }
    }

    bool check_completion(uint64_t wr_id) {
        poll();
        auto it = completed_ids_.find(wr_id);
        if (it != completed_ids_.end()) {
            completed_ids_.erase(it); // 关键优化：用完即删，防止 Set 膨胀
            return true;
        }
        return false;
    }

    friend class RDMARequest;

private:
    int rank_;
    int nRanks_;
    std::string root_ip_;
    struct ibv_context* ctx_ = nullptr;
    struct ibv_pd* pd_ = nullptr;
    struct ibv_cq* cq_ = nullptr;
    std::map<int, struct ibv_qp*> qps_;
    
    uint64_t next_wr_id_ = 0;
    std::unordered_set<uint64_t> completed_ids_;

    // --- 内存池数据结构 ---
    std::vector<RDMARequest> request_pool_; 
    std::vector<int> free_indices_;

    // --- 辅助函数 (保持不变) ---
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
        cq_ = ibv_create_cq(ctx_, 1024, nullptr, nullptr, 0); 
        ibv_free_device_list(dev_list);
    }

    void create_qps() {
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) continue;
            struct ibv_qp_init_attr attr = {};
            attr.send_cq = cq_;
            attr.recv_cq = cq_;
            attr.qp_type = IBV_QPT_RC;
            attr.cap.max_send_wr = 1024;
            attr.cap.max_recv_wr = 1024;
            attr.cap.max_send_sge = 1;
            attr.cap.max_recv_sge = 1;
            qps_[i] = ibv_create_qp(pd_, &attr);
        }
    }

    void exchange_and_connect() {
        struct ibv_port_attr port_attr;
        ibv_query_port(ctx_, 1, &port_attr);
        union ibv_gid my_gid;
        ibv_query_gid(ctx_, 1, 1, &my_gid);
        std::vector<RdmaInfo> my_infos;
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) my_infos.push_back({rank_, 0, 0, {0}});
            else {
                my_infos.push_back({rank_, qps_[i]->qp_num, port_attr.lid, {0}});
                memcpy(my_infos.back().gid, my_gid.raw, 16);
            }
        }
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
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) continue;
            connect_qp(qps_[i], global_registry[i][rank_]);
        }
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

inline void RDMARequest::wait() {
    while (!completed_) {
        if (transport_->check_completion(wr_id_)) {
            completed_ = true;
        }
    }
}

inline void RDMARequest::release() {
    transport_->freeRequest(pool_idx_);
}

} // namespace mini_nccl