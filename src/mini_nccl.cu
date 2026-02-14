#include "mini_nccl.h"
#include "transport/RDMATransport.h"
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat> 

namespace mini_nccl {

const size_t SLICE_SIZE = 128 * 1024; 

// =============================================================
// GPU Kernels
// =============================================================

// 1. 等待信号 (Polling)
__global__ void wait_kernel(volatile uint32_t* flag_addr, uint32_t expected) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (*flag_addr < expected) {
            // Spin wait
        }
    }
    __syncthreads();
}

// 2. 发送信号 (IPC Write)
// Phase 7 新增：通过 IPC 指针直接修改远程 Flag
__global__ void set_flag_kernel(volatile uint32_t* flag_addr, uint32_t val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *flag_addr = val;
    }
}

// 3. 计算算子
template<typename T> struct OpSum { __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; } };
template<typename T> struct OpProd { __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; } };
template<typename T> struct OpMax { __device__ __forceinline__ T operator()(const T& a, const T& b) const { return (a > b) ? a : b; } };
template<typename T> struct OpMin { __device__ __forceinline__ T operator()(const T& a, const T& b) const { return (a < b) ? a : b; } };

template<typename T, typename Op>
__global__ void elementwise_reduce_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, int n, Op op) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = op(a[i], b[i]);
}

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

// =============================================================
// Phase 7: Topology-Aware All-Reduce
// =============================================================
template<typename T, typename Op>
void allreduce_impl(T* data, int count, Op op, std::shared_ptr<Context> ctx, cudaStream_t stream) {
    int rank = ctx->rank();
    int size = ctx->size();
    if (size == 1) return;

    size_t type_size = sizeof(T);
    int chunk_count = count / size;
    size_t chunk_bytes = chunk_count * type_size;
    int send_to   = (rank + 1) % size;

    auto transport = std::dynamic_pointer_cast<RDMATransport>(ctx->transport());
    if (!transport) throw std::runtime_error("Phase 7 requires RDMATransport");

    auto mr_data = ctx->registerMemory(data, count * type_size);

    // 1. 分配双缓冲
    T* buffers[2];
    std::shared_ptr<MemoryRegion> mr_buffers[2];
    for(int i=0; i<2; ++i) {
        checkCuda(cudaHostAlloc(&buffers[i], SLICE_SIZE, cudaHostAllocDefault), "Alloc Buffers");
        mr_buffers[i] = ctx->registerMemory(buffers[i], SLICE_SIZE);
    }

    // 2. 设置本地信息
    transport->set_local_mem_info(
        (uint64_t)mr_data->ptr(), mr_data->rkey(),
        (uint64_t)mr_buffers[0]->ptr(), mr_buffers[0]->rkey(),
        (uint64_t)mr_buffers[1]->ptr(), mr_buffers[1]->rkey(),
        data, buffers[0], buffers[1] // Phase 7: 传入原始指针用于 IPC
    );

    // 3. 初始化连接 (交换 Hash 和 IPC Handle)
    transport->init(); 

    // 获取发送目标的 RDMA 信息和 IPC 指针
    auto send_peer_info = transport->get_peer_info(send_to);
    auto send_peer_ipc  = transport->get_peer_ipc_ptrs(send_to); // Phase 7

    uint32_t signal_seq = 1; 
    volatile uint32_t* d_flags = transport->get_flags_ptr();

    // ================= Phase 1: Scatter-Reduce =================
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + size) % size; 
        int recv_idx = (rank - i - 1 + size) % size; 
        size_t block_send_offset = send_idx * chunk_bytes;
        int num_slices = (chunk_bytes + SLICE_SIZE - 1) / SLICE_SIZE;
        
        for (int s = 0; s < num_slices; ++s) {
            size_t current_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - s * SLICE_SIZE);
            int current_slice_elems = current_slice_bytes / type_size;
            int curr_buff_idx = s % 2; 
            
            // 1. [Wait] 等待数据
            wait_kernel<<<1, 1, 0, stream>>>(&d_flags[curr_buff_idx], signal_seq);

            // 2. [Compute] 计算
            T* d_target = data + (recv_idx * chunk_count) + (s * (SLICE_SIZE/type_size));
            int threads = 256;
            int blocks = (current_slice_elems + threads - 1) / threads;
            elementwise_reduce_kernel<<<blocks, threads, 0, stream>>>(d_target, buffers[curr_buff_idx], d_target, current_slice_elems, op);

            // 3. [Push] 发送给右边 (Phase 7 分流)
            if (send_peer_ipc.is_local) {
                // --- IPC Path (同机极速通道) ---
                
                // 3.1 直接 Copy 到对方 Buffer
                void* remote_buffer = send_peer_ipc.buffer_ptr[s % 2];
                // 计算源地址：data + offset
                void* local_src = (char*)data + block_send_offset + s * SLICE_SIZE;
                
                checkCuda(cudaMemcpyAsync(remote_buffer, local_src, current_slice_bytes, cudaMemcpyDeviceToDevice, stream), "IPC Memcpy");

                // 3.2 直接写对方 Flag
                // 注意：flag 指针是 uint32_t*，需要偏移 curr_buff_idx
                volatile uint32_t* remote_flag = (volatile uint32_t*)send_peer_ipc.flag_ptr + curr_buff_idx;
                set_flag_kernel<<<1, 1, 0, stream>>>(remote_flag, signal_seq);

            } else {
                // --- RDMA Path (跨机常规通道) ---
                
                // 3.1 RDMA Write Buffer
                uint64_t remote_dst_addr = send_peer_info.buffer_addr[s % 2];
                uint32_t remote_rkey = send_peer_info.buffer_rkey[s % 2];
                Request* req_write = transport->write(send_to, mr_data, block_send_offset + s * SLICE_SIZE, current_slice_bytes, 
                                                    remote_dst_addr, remote_rkey);
                
                // 3.2 RDMA Write Signal
                Request* req_sig = transport->write_signal(send_to, curr_buff_idx, signal_seq);

                // 简单的流控回收
                if (s % 16 == 0) {
                    req_write->wait(); req_write->release();
                    req_sig->wait(); req_sig->release();
                } else {
                    req_write->wait(); req_write->release();
                    req_sig->wait(); req_sig->release();
                }
            }

            signal_seq++; 
        }
    }
    
    // ================= Phase 2: All-Gather =================
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + 1 + size) % size;
        size_t block_send_offset = send_idx * chunk_bytes;
        int num_slices = (chunk_bytes + SLICE_SIZE - 1) / SLICE_SIZE;

        for (int s = 0; s < num_slices; ++s) {
             size_t current_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - s * SLICE_SIZE);
             int curr_flag_idx = s % 2;
             
             // 1. [Wait]
             wait_kernel<<<1, 1, 0, stream>>>(&d_flags[curr_flag_idx], signal_seq);
             
             // 2. [Push]
             if (send_peer_ipc.is_local) {
                 // --- IPC Path ---
                 
                 // 2.1 直接 Copy 到对方 Final Data
                 // 注意：send_peer_ipc.data_ptr 是对方 data 的基地址，需要加上偏移
                 void* remote_dst = (char*)send_peer_ipc.data_ptr + block_send_offset + s * SLICE_SIZE;
                 // 源数据在本地 data 的 block_send_offset 处
                 void* local_src = (char*)data + block_send_offset + s * SLICE_SIZE;
                 
                 checkCuda(cudaMemcpyAsync(remote_dst, local_src, current_slice_bytes, cudaMemcpyDeviceToDevice, stream), "IPC Memcpy Phase2");
                 
                 // 2.2 写 Signal
                 volatile uint32_t* remote_flag = (volatile uint32_t*)send_peer_ipc.flag_ptr + curr_flag_idx;
                 set_flag_kernel<<<1, 1, 0, stream>>>(remote_flag, signal_seq);
                 
             } else {
                 // --- RDMA Path ---
                 uint64_t remote_dst_addr = send_peer_info.data_addr + block_send_offset + s * SLICE_SIZE;
                 Request* req_write = transport->write(send_to, mr_data, block_send_offset + s * SLICE_SIZE, current_slice_bytes,
                                                     remote_dst_addr, send_peer_info.data_rkey);
                 
                 Request* req_sig = transport->write_signal(send_to, curr_flag_idx, signal_seq);
                 
                 req_write->wait(); req_write->release();
                 req_sig->wait(); req_sig->release();
             }
             
             signal_seq++;
        }
    }

    checkCuda(cudaStreamSynchronize(stream), "Stream Sync");
    for(int i=0; i<2; ++i) cudaFreeHost(buffers[i]);
}

#define DISPATCH_OP(TYPE_T, TYPE_ENUM, CTX, STREAM) \
    switch(op) { \
        case RedOp::Sum:  allreduce_impl(static_cast<TYPE_T*>(data), count, OpSum<TYPE_T>(), CTX, STREAM); break; \
        case RedOp::Prod: allreduce_impl(static_cast<TYPE_T*>(data), count, OpProd<TYPE_T>(), CTX, STREAM); break; \
        case RedOp::Max:  allreduce_impl(static_cast<TYPE_T*>(data), count, OpMax<TYPE_T>(), CTX, STREAM); break; \
        case RedOp::Min:  allreduce_impl(static_cast<TYPE_T*>(data), count, OpMin<TYPE_T>(), CTX, STREAM); break; \
    }

void allreduce(void* data, int count, DataType dtype, RedOp op, std::shared_ptr<Context> ctx, cudaStream_t stream) {
    switch(dtype) {
        case DataType::Float32: DISPATCH_OP(float, dtype, ctx, stream); break;
        case DataType::Float64: DISPATCH_OP(double, dtype, ctx, stream); break;
        case DataType::Int32:   DISPATCH_OP(int, dtype, ctx, stream); break;
        default: std::cerr << "Unknown DataType!" << std::endl; exit(1);
    }
}

} // namespace mini_nccl