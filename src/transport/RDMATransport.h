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
#include <unistd.h> 
#include <atomic> 
#include <string>

namespace mini_nccl {

struct DynamicMemInfo {
    int rank;
    uint64_t data_addr; uint32_t data_rkey;
    uint64_t buf0_addr; uint32_t buf0_rkey;
    uint64_t buf1_addr; uint32_t buf1_rkey;
    cudaIpcMemHandle_t data_ipc;
    cudaIpcMemHandle_t buf0_ipc;
    cudaIpcMemHandle_t buf1_ipc;
};

struct RdmaInfo {
    int rank;
    uint32_t qp_num;
    uint16_t lid;
    uint8_t gid[16];
    uint64_t flag_addr; uint32_t flag_rkey;
    uint64_t host_hash;             
    cudaIpcMemHandle_t flag_ipc;    
    uint64_t data_addr; uint32_t data_rkey;
    uint64_t buffer_addr[2]; uint32_t buffer_rkey[2];
    cudaIpcMemHandle_t data_ipc;    
    cudaIpcMemHandle_t buffer_ipc[2];
};

class RDMAMemoryRegion : public MemoryRegion {
public:
    // 核心修改：支持 Device Pointer 注册 (GPUDirect)
    RDMAMemoryRegion(struct ibv_pd* pd, void* ptr, size_t size) : ptr_(ptr), size_(size) {
        // 尝试注册 MR。如果是 GPU 显存且系统支持 GPUDirect (nvidia-peermem)，这里会成功。
        // 如果是 Host 内存，这里也会成功。
        mr_ = ibv_reg_mr(pd, ptr, size, 
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        
        if (!mr_) {
            // 如果注册失败，抛出详细错误。
            // 在 WSL2 上注册 Device Pointer 通常会失败，因为缺少内核模块。
            throw std::runtime_error("Failed to register MR (GPUDirect RDMA not supported or memory not pinned?)");
        }
    }
    ~RDMAMemoryRegion() { if (mr_) ibv_dereg_mr(mr_); }
    void* ptr() const override { return ptr_; }
    size_t size() const override { return size_; }
    uint32_t rkey() const override { return mr_->rkey; } 
    uint32_t lkey() const { return mr_->lkey; }
private:
    void* ptr_;
    size_t size_;
    struct ibv_mr* mr_;
};

class RDMATransport;

class RDMARequest : public Request {
public:
    RDMARequest() : transport_(nullptr), wr_id_(0), completed_(false), pool_idx_(-1) {}
    void reset(RDMATransport* transport, uint64_t wr_id, int pool_idx) {
        transport_ = transport; wr_id_ = wr_id; pool_idx_ = pool_idx; completed_ = false;
    }
    void wait() override; 
    void release() override;
    bool isCompleted() const override { return completed_; }
    void markCompleted() { completed_ = true; }
    uint64_t id() const { return wr_id_; }
private:
    RDMATransport* transport_;
    uint64_t wr_id_;
    volatile bool completed_;
    int pool_idx_; 
};

struct PeerIpcPtrs {
    bool is_local = false;
    void* flag_ptr = nullptr;
    void* data_ptr = nullptr;
    void* buffer_ptr[2] = {nullptr, nullptr};
};

class RDMATransport : public Transport {
public:
    RDMATransport(int rank, int nRanks, std::string root_ip = "127.0.0.1") 
        : rank_(rank), nRanks_(nRanks), root_ip_(root_ip) {

        cudaSetDeviceFlags(cudaDeviceMapHost);
        
        setup_device();
        init_memory_pool();
        
        char hostname[1024];
        gethostname(hostname, 1024);
        my_host_hash_ = std::hash<std::string>{}(std::string(hostname));
        
        if (cudaHostAlloc(&abort_flag_, sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess)
            throw std::runtime_error("Failed to alloc abort_flag");
        *abort_flag_ = 0; 

        size_t flag_size = 1024 * sizeof(uint32_t);
        if (cudaHostAlloc(&flags_, flag_size, cudaHostAllocMapped) != cudaSuccess)
            throw std::runtime_error("Failed to alloc flags");
            
        mr_flags_ = ibv_reg_mr(pd_, flags_, flag_size, 
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        memset(flags_, 0, flag_size);
    }

    ~RDMATransport() {
        if (mr_flags_) ibv_dereg_mr(mr_flags_);
        if (flags_) cudaFreeHost(flags_);
        if (abort_flag_) cudaFreeHost(abort_flag_); 
        for (auto& pair : qps_) if (pair.second) ibv_destroy_qp(pair.second);
        if (cq_) ibv_destroy_cq(cq_);
        if (pd_) ibv_dealloc_pd(pd_);
        if (ctx_) ibv_close_device(ctx_);
        
        for (auto& pair : peer_ipc_ptrs_) {
            if (pair.second.is_local) {
                if (pair.second.flag_ptr) cudaIpcCloseMemHandle(pair.second.flag_ptr);
                if (pair.second.data_ptr) cudaIpcCloseMemHandle(pair.second.data_ptr);
                if (pair.second.buffer_ptr[0]) cudaIpcCloseMemHandle(pair.second.buffer_ptr[0]);
                if (pair.second.buffer_ptr[1]) cudaIpcCloseMemHandle(pair.second.buffer_ptr[1]);
            }
        }
        cudaGetLastError(); 
    }

    void init() override {
        create_qps();
        exchange_static_info_and_connect(); 
    }

    uint32_t* get_flags_ptr() override { 
        uint32_t* d_ptr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(&d_ptr, flags_, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error("get_flags_ptr failed: " + std::string(cudaGetErrorString(err)));
        }
        return d_ptr;
    }

    uint32_t* get_abort_flag_dev_ptr() {
        uint32_t* d_ptr = nullptr;
        cudaHostGetDevicePointer(&d_ptr, abort_flag_, 0);
        return d_ptr;
    }

    void abort() { *abort_flag_ = 1; }

    RdmaInfo get_peer_info(int rank) { return peer_infos_[rank]; }
    PeerIpcPtrs get_peer_ipc_ptrs(int rank) { return peer_ipc_ptrs_[rank]; }

    void exchange_dynamic_info(uint64_t data_addr, uint32_t data_rkey,
                               uint64_t buf0_addr, uint32_t buf0_rkey,
                               uint64_t buf1_addr, uint32_t buf1_rkey,
                               void* raw_data_ptr, void* raw_buf0_ptr, void* raw_buf1_ptr) {
        
        DynamicMemInfo my_dyn;
        my_dyn.rank = rank_;
        my_dyn.data_addr = data_addr; my_dyn.data_rkey = data_rkey;
        my_dyn.buf0_addr = buf0_addr; my_dyn.buf0_rkey = buf0_rkey;
        my_dyn.buf1_addr = buf1_addr; my_dyn.buf1_rkey = buf1_rkey;
        
        void* d_ptr = nullptr;
        cudaError_t ret = cudaSuccess;

        // 尝试获取 IPC Handle。注意：如果是 GPUDirect 模式 (raw_data_ptr 是 Device 内存)，
        // 这里依然可以尝试获取 IPC Handle。这与 RDMA 路径不冲突。
        
        ret = cudaHostGetDevicePointer(&d_ptr, raw_data_ptr, 0);
        // 如果是 Device Pointer，GetDevicePointer 可能会失败或返回自身。
        // 我们改为直接用 raw_ptr 尝试获取 IPC Handle
        if (ret != cudaSuccess) { 
            cudaGetLastError(); // 并不是 Pinned Host Memory，可能是 Device Memory
            d_ptr = raw_data_ptr;
        }
        if (cudaIpcGetMemHandle(&my_dyn.data_ipc, d_ptr) != cudaSuccess) {
             cudaGetLastError(); memset(&my_dyn.data_ipc, 0, sizeof(my_dyn.data_ipc)); 
        }

        // Buffer 0/1 也是同理
        ret = cudaHostGetDevicePointer(&d_ptr, raw_buf0_ptr, 0);
        if (ret != cudaSuccess) { cudaGetLastError(); d_ptr = raw_buf0_ptr; }
        if (cudaIpcGetMemHandle(&my_dyn.buf0_ipc, d_ptr) != cudaSuccess) {
             cudaGetLastError(); memset(&my_dyn.buf0_ipc, 0, sizeof(my_dyn.buf0_ipc)); 
        }

        ret = cudaHostGetDevicePointer(&d_ptr, raw_buf1_ptr, 0);
        if (ret != cudaSuccess) { cudaGetLastError(); d_ptr = raw_buf1_ptr; }
        if (cudaIpcGetMemHandle(&my_dyn.buf1_ipc, d_ptr) != cudaSuccess) {
             cudaGetLastError(); memset(&my_dyn.buf1_ipc, 0, sizeof(my_dyn.buf1_ipc)); 
        }

        std::vector<DynamicMemInfo> all_dyn(nRanks_);
        if (rank_ == 0) {
            all_dyn[0] = my_dyn;
            for (auto& sock : client_sockets_) {
                DynamicMemInfo info;
                sock->recv(&info, sizeof(DynamicMemInfo));
                all_dyn[info.rank] = info;
            }
            for (auto& sock : client_sockets_) {
                sock->send(all_dyn.data(), nRanks_ * sizeof(DynamicMemInfo));
            }
        } else {
            root_socket_->send(&my_dyn, sizeof(DynamicMemInfo));
            root_socket_->recv(all_dyn.data(), nRanks_ * sizeof(DynamicMemInfo));
        }

        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) continue;
            peer_infos_[i].data_addr = all_dyn[i].data_addr;
            peer_infos_[i].data_rkey = all_dyn[i].data_rkey;
            peer_infos_[i].buffer_addr[0] = all_dyn[i].buf0_addr;
            peer_infos_[i].buffer_rkey[0] = all_dyn[i].buf0_rkey;
            peer_infos_[i].buffer_addr[1] = all_dyn[i].buf1_addr;
            peer_infos_[i].buffer_rkey[1] = all_dyn[i].buf1_rkey;
            
            if (peer_ipc_ptrs_[i].is_local) {
                if (peer_ipc_ptrs_[i].data_ptr) { cudaIpcCloseMemHandle(peer_ipc_ptrs_[i].data_ptr); peer_ipc_ptrs_[i].data_ptr = nullptr; }
                if (peer_ipc_ptrs_[i].buffer_ptr[0]) { cudaIpcCloseMemHandle(peer_ipc_ptrs_[i].buffer_ptr[0]); peer_ipc_ptrs_[i].buffer_ptr[0] = nullptr; }
                if (peer_ipc_ptrs_[i].buffer_ptr[1]) { cudaIpcCloseMemHandle(peer_ipc_ptrs_[i].buffer_ptr[1]); peer_ipc_ptrs_[i].buffer_ptr[1] = nullptr; }
                
                cudaGetLastError();

                cudaError_t ipc_err = cudaSuccess;
                void* d_data = nullptr;
                void* d_buf0 = nullptr;
                void* d_buf1 = nullptr;

                if(ipc_err == cudaSuccess) ipc_err = cudaIpcOpenMemHandle(&d_data, all_dyn[i].data_ipc, cudaIpcMemLazyEnablePeerAccess);
                if(ipc_err == cudaSuccess) ipc_err = cudaIpcOpenMemHandle(&d_buf0, all_dyn[i].buf0_ipc, cudaIpcMemLazyEnablePeerAccess);
                if(ipc_err == cudaSuccess) ipc_err = cudaIpcOpenMemHandle(&d_buf1, all_dyn[i].buf1_ipc, cudaIpcMemLazyEnablePeerAccess);

                if (ipc_err == cudaSuccess) {
                    peer_ipc_ptrs_[i].data_ptr = d_data;
                    peer_ipc_ptrs_[i].buffer_ptr[0] = d_buf0;
                    peer_ipc_ptrs_[i].buffer_ptr[1] = d_buf1;
                } else {
                    cudaGetLastError(); 
                    peer_ipc_ptrs_[i].is_local = false; 
                }
            }
        }
    }

    Request* write(int rank, std::shared_ptr<MemoryRegion> local_mr, size_t offset, size_t length,
                   uint64_t remote_addr, uint32_t remote_rkey, bool signaled = true) override {
        auto rmr = std::static_pointer_cast<RDMAMemoryRegion>(local_mr);
        uint64_t wr_id = next_wr_id_++; 
        struct ibv_sge sge;
        sge.addr = (uint64_t)rmr->ptr() + offset;
        sge.length = length;
        sge.lkey = rmr->lkey();

        struct ibv_send_wr wr = {};
        wr.wr_id = wr_id;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = 0;
        if (signaled) wr.send_flags |= IBV_SEND_SIGNALED; 
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = remote_rkey;

        struct ibv_send_wr* bad_wr;
        if (ibv_post_send(qps_[rank], &wr, &bad_wr)) throw std::runtime_error("ibv_post_send (WRITE) failed");
        
        if (!signaled) return nullptr;
        return allocateRequest(wr_id);
    }

    Request* write_signal(int rank, int flag_idx, uint32_t value, bool signaled = true) override {
        uint64_t wr_id = next_wr_id_++;
        uint64_t remote_addr = peer_infos_[rank].flag_addr + flag_idx * sizeof(uint32_t);
        uint32_t rkey = peer_infos_[rank].flag_rkey;

        struct ibv_send_wr wr = {};
        wr.wr_id = wr_id;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_INLINE; 
        if (signaled) wr.send_flags |= IBV_SEND_SIGNALED;
        
        struct ibv_sge sge;
        sge.addr = (uint64_t)&value; 
        sge.length = sizeof(uint32_t);
        sge.lkey = 0; 

        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        struct ibv_send_wr* bad_wr;
        if (ibv_post_send(qps_[rank], &wr, &bad_wr)) throw std::runtime_error("ibv_post_send (SIGNAL) failed");
        
        if (!signaled) return nullptr;
        return allocateRequest(wr_id);
    }

    Request* isend(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) override { return nullptr; }
    Request* irecv(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) override { return nullptr; }
    
    // 覆盖：自动检测指针类型并注册
    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) override {
        return std::make_shared<RDMAMemoryRegion>(pd_, ptr, size);
    }

    RDMARequest* allocateRequest(uint64_t wr_id) {
        if (free_indices_.empty()) return nullptr;
        int idx = free_indices_.back(); free_indices_.pop_back();
        RDMARequest* req = &request_pool_[idx];
        req->reset(this, wr_id, idx);
        return req;
    }
    void freeRequest(int idx) { free_indices_.push_back(idx); }
    void poll() {
        struct ibv_wc wc[16];
        int n = ibv_poll_cq(cq_, 16, wc);
        for (int i = 0; i < n; ++i) completed_ids_.insert(wc[i].wr_id);
    }
    bool check_completion(uint64_t wr_id) {
        poll();
        auto it = completed_ids_.find(wr_id);
        if (it != completed_ids_.end()) { completed_ids_.erase(it); return true; }
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
    
    std::shared_ptr<Socket> root_socket_; 
    std::vector<std::shared_ptr<Socket>> client_sockets_; 

    uint32_t* flags_ = nullptr;
    uint32_t* abort_flag_ = nullptr; 
    struct ibv_mr* mr_flags_ = nullptr;
    
    std::map<int, RdmaInfo> peer_infos_; 
    std::map<int, PeerIpcPtrs> peer_ipc_ptrs_;

    uint64_t my_host_hash_;
    cudaIpcMemHandle_t my_flag_ipc_;

    uint64_t next_wr_id_ = 0;
    std::unordered_set<uint64_t> completed_ids_;
    std::vector<RDMARequest> request_pool_; 
    std::vector<int> free_indices_;

    void init_memory_pool() {
        int pool_size = 4096; 
        request_pool_.resize(pool_size);
        free_indices_.reserve(pool_size);
        for (int i = 0; i < pool_size; ++i) free_indices_.push_back(i);
    }

    void setup_device() {
        int num_devices;
        struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
        if(!dev_list) throw std::runtime_error("No RDMA device");
        struct ibv_device* device = nullptr;
        for (int i = 0; i < num_devices; ++i) {
            if (std::string(ibv_get_device_name(dev_list[i])) == "rxe0") {
                device = dev_list[i]; break;
            }
        }
        if(!device) throw std::runtime_error("rxe0 not found");
        ctx_ = ibv_open_device(device);
        pd_ = ibv_alloc_pd(ctx_);
        cq_ = ibv_create_cq(ctx_, 1024, nullptr, nullptr, 0); 
        ibv_free_device_list(dev_list);
    }

    void create_qps() {
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) continue;
            struct ibv_qp_init_attr attr = {};
            attr.send_cq = cq_; attr.recv_cq = cq_; attr.qp_type = IBV_QPT_RC;
            attr.cap.max_send_wr = 1024; attr.cap.max_recv_wr = 1024;
            attr.cap.max_send_sge = 1; attr.cap.max_recv_sge = 1;
            attr.cap.max_inline_data = 64; 
            qps_[i] = ibv_create_qp(pd_, &attr);
        }
    }

    void exchange_static_info_and_connect() {
        struct ibv_port_attr port_attr;
        ibv_query_port(ctx_, 1, &port_attr);
        union ibv_gid my_gid;
        ibv_query_gid(ctx_, 1, 1, &my_gid);
        
        void* d_flag; 
        if (cudaHostGetDevicePointer(&d_flag, flags_, 0) != cudaSuccess)
            throw std::runtime_error("Failed to get device ptr for flags");
            
        if (cudaIpcGetMemHandle(&my_flag_ipc_, d_flag) != cudaSuccess) {
            cudaGetLastError(); 
            memset(&my_flag_ipc_, 0, sizeof(my_flag_ipc_)); 
        }

        std::vector<RdmaInfo> my_infos;
        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) {
                my_infos.push_back({}); 
            } else {
                RdmaInfo info = {}; 
                info.rank = rank_;
                info.qp_num = qps_[i]->qp_num;
                info.lid = port_attr.lid;
                memcpy(info.gid, my_gid.raw, 16);
                info.flag_addr = (uint64_t)flags_;
                info.flag_rkey = mr_flags_->rkey;
                info.host_hash = my_host_hash_;
                info.flag_ipc = my_flag_ipc_;
                my_infos.push_back(info);
            }
        }
        
        std::vector<std::vector<RdmaInfo>> global_registry(nRanks_);
        if (rank_ == 0) {
            ServerSocket server(8888);
            global_registry[0] = my_infos;
            for (int i = 1; i < nRanks_; ++i) {
                auto sock = server.accept();
                client_sockets_.push_back(sock); 
                int r; sock->recv(&r, sizeof(int));
                std::vector<RdmaInfo> peer_infos(nRanks_);
                sock->recv(peer_infos.data(), nRanks_ * sizeof(RdmaInfo));
                global_registry[r] = peer_infos;
            }
            for (auto& sock : client_sockets_) {
                for (int i = 0; i < nRanks_; ++i) 
                    sock->send(global_registry[i].data(), nRanks_ * sizeof(RdmaInfo));
            }
        } else {
            root_socket_ = connect_to(root_ip_, 8888); 
            root_socket_->send(&rank_, sizeof(int));
            root_socket_->send(my_infos.data(), nRanks_ * sizeof(RdmaInfo));
            for(int i=0; i<nRanks_; ++i) {
                global_registry[i].resize(nRanks_);
                root_socket_->recv(global_registry[i].data(), nRanks_ * sizeof(RdmaInfo));
            }
        }

        for (int i = 0; i < nRanks_; ++i) {
            if (i == rank_) continue;
            peer_infos_[i] = global_registry[i][rank_]; 
            connect_qp(qps_[i], peer_infos_[i]);

            PeerIpcPtrs ipc_ptrs;
            ipc_ptrs.is_local = false;

            if (peer_infos_[i].host_hash == my_host_hash_) {
                cudaError_t err = cudaIpcOpenMemHandle(&ipc_ptrs.flag_ptr, peer_infos_[i].flag_ipc, cudaIpcMemLazyEnablePeerAccess);
                if(err == cudaSuccess) {
                    ipc_ptrs.is_local = true;
                } else {
                    cudaGetLastError(); 
                }
            }
            peer_ipc_ptrs_[i] = ipc_ptrs;
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
        if (transport_->check_completion(wr_id_)) completed_ = true;
    }
}
inline void RDMARequest::release() {
    transport_->freeRequest(pool_idx_);
}

} // namespace mini_nccl