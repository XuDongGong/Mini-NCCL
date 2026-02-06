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

// RDMA 请求对象：用于追踪 isend/irecv 的状态
class RDMARequest : public Request {
public:
    // 默认构造，用于 vector 初始化
    RDMARequest() : transport_(nullptr), wr_id_(0), completed_(false), pool_idx_(-1) {}

    // 初始化方法 (替代构造函数)
    void reset(RDMATransport* transport, uint64_t wr_id, int pool_idx) {
        transport_ = transport;
        wr_id_ = wr_id;
        pool_idx_ = pool_idx;
        completed_ = false;
    }

    void wait() override; 
    
    // 归还自己
    void release() override;

    bool isCompleted() const override { return completed_; }
    void markCompleted() { completed_ = true; }
    uint64_t id() const { return wr_id_; }

private:
    RDMATransport* transport_;
    uint64_t wr_id_;
    volatile bool completed_;
    int pool_idx_; // 记录自己在 pool 中的位置，方便归还
};

class RDMATransport : public Transport {
public:
RDMATransport(int rank, int nRanks, std::string root_ip = "127.0.0.1") 
        : rank_(rank), nRanks_(nRanks), root_ip_(root_ip) {
        setup_device();
        
        // --- 预分配内存池 ---
        // 预分配 1024 个请求对象，足够双缓冲流水线跑满
        // 实际 NCCL 会动态扩容，这里简化为固定大小
        int pool_size = 10240; 
        request_pool_.resize(pool_size);
        free_indices_.reserve(pool_size);
        for (int i = 0; i < pool_size; ++i) {
            free_indices_.push_back(i);
        }
    }

    ~RDMATransport() {
        // ... (清理逻辑保持不变) ...
        for (auto& pair : qps_) if (pair.second) ibv_destroy_qp(pair.second);
        if (cq_) ibv_destroy_cq(cq_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (ctx_) ibv_close_device(ctx_);
    }

    void init() override {
        create_qps();
        exchange_and_connect();
    }

    // --- 对象池分配逻辑 ---
    RDMARequest* allocateRequest(uint64_t wr_id) {
        if (free_indices_.empty()) {
            throw std::runtime_error("Request Pool Exhausted! (Circular buffer full)");
        }
        int idx = free_indices_.back();
        free_indices_.pop_back();

        RDMARequest* req = &request_pool_[idx];
        req->reset(this, wr_id, idx);
        return req;
    }

    // --- 对象池回收逻辑 ---
    void freeRequest(int idx) {
        // 简单压栈
        free_indices_.push_back(idx);
    }

    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) override {
        return std::make_shared<RDMAMemoryRegion>(pd_, ptr, size);
    }

    // --- isend (零分配版) ---
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

        // 从池中拿，而不是 new
        return allocateRequest(wr_id);
    }

    // --- irecv (零分配版) ---
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
        if (completed_ids_.count(wr_id)) {
            // 优化：一旦确认完成，立刻从 set 中移除，防止 set 无限膨胀
            // (这是 v1.3.1 的一个小优化)
            completed_ids_.erase(wr_id);
            return true;
        }
        return false;
    }

    // 友元声明，允许 Request 访问 freeRequest
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
    std::unordered_set<uint64_t> completed_ids_; // 已完成的任务ID池

    // --- 👇 内存池数据结构 ---
    std::vector<RDMARequest> request_pool_; // 连续内存块，Cache 友好
    std::vector<int> free_indices_;         // 空闲栈

    // ... (保留 setup_device, create_qp, connect_qp 等私有辅助函数) ...
    // 为了代码简洁，请把上一步写好的辅助函数都贴在这里
    // 必须保留的辅助函数：setup_device, create_qp, connect_qp
    
    // --- 辅助函数重新粘贴区 (方便你复制) ---
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
        // ... 把上一步 init() 中从 "准备好我自己的名片" 到 "End" 的代码贴在这里 ...
        // 由于篇幅限制，这里用伪代码指代，请务必把上一版 init() 里的 TCP 握手逻辑拷过来
        // 如果需要我完整重写这一段，请告诉我。
        // 简单策略：直接把上一步 init() 函数体里的内容，除了第一步 create_qp，剩下的全放这里。
        
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

inline void RDMARequest::release() {
    transport_->freeRequest(pool_idx_);
}

} // namespace mini_nccl