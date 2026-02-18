#include "mini_nccl.h"
#include "transport/RDMATransport.h"
#include "Config.h"
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat> 
#include <chrono> 
#include <thread> 
#include <queue> 
#include <string>

namespace mini_nccl {

// >>> 🚀 提升五: Context 构造函数实现 >>>
Context::Context(int rank, int size, std::shared_ptr<Transport> transport)
    : rank_(rank), size_(size), transport_(transport) {
    // 在 Context 初始化时，直接根据配置预分配内存
    // 这样在 allreduce 热路径中就不需要反复 malloc/free 了
    size_t slice = Config::getInstance().slice_size;
    allocate_scratch_buffer(slice);
}
// <<< 提升五结束 <<<

__global__ void wait_kernel(volatile uint32_t* flag_addr, uint32_t expected, volatile uint32_t* abort_flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (*flag_addr < expected) {
            // GPU-Side Polling
            if (*abort_flag != 0) return;
        }
    }
    __syncthreads();
}

__global__ void set_flag_kernel(volatile uint32_t* flag_addr, uint32_t val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *flag_addr = val;
    }
}

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
        std::string err_str = "CUDA Error: " + std::string(msg) + " : " + std::string(cudaGetErrorString(result));
        throw std::runtime_error(err_str);
    }
}

template<typename T, typename Op>
void allreduce_impl(T* data, int count, Op op, std::shared_ptr<Context> ctx, cudaStream_t stream) {
    // 获取配置
    auto& cfg = Config::getInstance();
    const size_t SLICE_SIZE = cfg.slice_size;
    const int WINDOW_SIZE = cfg.window_size;
    const int SIGNAL_BATCH = cfg.signal_batch;

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

    T* buffers[2];
    std::shared_ptr<MemoryRegion> mr_buffers[2];
    
    // >>> 🚀 提升五: 使用复用 Buffer >>>
    // 移除旧的 cudaHostAlloc 代码，改用 get_scratch_buffer
    for(int i=0; i<2; ++i) {
        // 直接从 Context 获取，零开销 (Zero Overhead)
        buffers[i] = (T*)ctx->get_scratch_buffer(i);
        if (!buffers[i]) throw std::runtime_error("Scratch buffer not allocated or invalid index");
        
        // 注册 MR (配合 MR Cache，这也将是零开销)
        mr_buffers[i] = ctx->registerMemory(buffers[i], SLICE_SIZE);
    }
    // <<< 提升五结束 <<<

    transport->exchange_dynamic_info(
        (uint64_t)mr_data->ptr(), mr_data->rkey(),
        (uint64_t)mr_buffers[0]->ptr(), mr_buffers[0]->rkey(),
        (uint64_t)mr_buffers[1]->ptr(), mr_buffers[1]->rkey(),
        data, buffers[0], buffers[1]
    );

    auto send_peer_info = transport->get_peer_info(send_to);
    auto send_peer_ipc  = transport->get_peer_ipc_ptrs(send_to);

    uint32_t signal_seq = 1; 
    volatile uint32_t* d_flags = transport->get_flags_ptr();
    volatile uint32_t* d_abort_flag = transport->get_abort_flag_dev_ptr();

    std::queue<Request*> pending_reqs;

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
            
            bool do_signal = ((s % SIGNAL_BATCH) == 0) || (s == num_slices - 1);

            wait_kernel<<<1, 1, 0, stream>>>(&d_flags[curr_buff_idx], signal_seq, d_abort_flag);

            T* d_target = data + (recv_idx * chunk_count) + (s * (SLICE_SIZE/type_size));
            int threads = 256;
            int blocks = (current_slice_elems + threads - 1) / threads;
            elementwise_reduce_kernel<<<blocks, threads, 0, stream>>>(d_target, buffers[curr_buff_idx], d_target, current_slice_elems, op);

            if (send_peer_ipc.is_local) {
                void* remote_buffer = send_peer_ipc.buffer_ptr[s % 2];
                void* local_src = (char*)data + block_send_offset + s * SLICE_SIZE;
                checkCuda(cudaMemcpyAsync(remote_buffer, local_src, current_slice_bytes, cudaMemcpyDeviceToDevice, stream), "IPC Memcpy");
                volatile uint32_t* remote_flag = (volatile uint32_t*)send_peer_ipc.flag_ptr + curr_buff_idx;
                set_flag_kernel<<<1, 1, 0, stream>>>(remote_flag, signal_seq);
            } else {
                uint64_t remote_dst_addr = send_peer_info.buffer_addr[s % 2];
                uint32_t remote_rkey = send_peer_info.buffer_rkey[s % 2];
                
                Request* req_write = transport->write(send_to, mr_data, block_send_offset + s * SLICE_SIZE, current_slice_bytes, 
                                                    remote_dst_addr, remote_rkey, false);
                
                Request* req_sig = transport->write_signal(send_to, curr_buff_idx, signal_seq, do_signal);
                
                if (req_sig) pending_reqs.push(req_sig);
                if (pending_reqs.size() > WINDOW_SIZE) {
                    Request* oldest = pending_reqs.front();
                    oldest->wait(); oldest->release();
                    pending_reqs.pop();
                }
            }
            signal_seq++; 
        }
    }
    
    // ================= Phase 2: All-Gather =================
    while(!pending_reqs.empty()) {
        pending_reqs.front()->wait(); pending_reqs.front()->release(); pending_reqs.pop();
    }

    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + 1 + size) % size;
        size_t block_send_offset = send_idx * chunk_bytes;
        int num_slices = (chunk_bytes + SLICE_SIZE - 1) / SLICE_SIZE;

        for (int s = 0; s < num_slices; ++s) {
             size_t current_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - s * SLICE_SIZE);
             int curr_flag_idx = s % 2;
             bool do_signal = ((s % SIGNAL_BATCH) == 0) || (s == num_slices - 1);
             
             wait_kernel<<<1, 1, 0, stream>>>(&d_flags[curr_flag_idx], signal_seq, d_abort_flag);
             
             if (send_peer_ipc.is_local) {
                 void* remote_dst = (char*)send_peer_ipc.data_ptr + block_send_offset + s * SLICE_SIZE;
                 void* local_src = (char*)data + block_send_offset + s * SLICE_SIZE;
                 checkCuda(cudaMemcpyAsync(remote_dst, local_src, current_slice_bytes, cudaMemcpyDeviceToDevice, stream), "IPC Memcpy Phase2");
                 volatile uint32_t* remote_flag = (volatile uint32_t*)send_peer_ipc.flag_ptr + curr_flag_idx;
                 set_flag_kernel<<<1, 1, 0, stream>>>(remote_flag, signal_seq);
             } else {
                 uint64_t remote_dst_addr = send_peer_info.data_addr + block_send_offset + s * SLICE_SIZE;
                 
                 transport->write(send_to, mr_data, block_send_offset + s * SLICE_SIZE, current_slice_bytes,
                                                     remote_dst_addr, send_peer_info.data_rkey, false);
                 
                 Request* req_sig = transport->write_signal(send_to, curr_flag_idx, signal_seq, do_signal);
                 
                 if (req_sig) pending_reqs.push(req_sig);
                 if (pending_reqs.size() > WINDOW_SIZE) {
                    Request* oldest = pending_reqs.front();
                    oldest->wait(); oldest->release();
                    pending_reqs.pop();
                 }
             }
             signal_seq++;
        }
    }

    while(!pending_reqs.empty()) {
        pending_reqs.front()->wait(); pending_reqs.front()->release(); pending_reqs.pop();
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    const double TIMEOUT_SECONDS = 10.0;

    while (cudaStreamQuery(stream) == cudaErrorNotReady) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;
        
        if (elapsed.count() > TIMEOUT_SECONDS) {
            std::cerr << "[Watchdog] TIMEOUT DETECTED! Aborting GPU kernels..." << std::endl;
            transport->abort();
            checkCuda(cudaStreamSynchronize(stream), "Stream Sync after Abort");
            throw std::runtime_error("NCCL Watchdog Timeout: Communication hang detected.");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    checkCuda(cudaGetLastError(), "Final Check");
    // 不再需要 cudaFreeHost(buffers[i])，因为它是 Context 管理的
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
        default: 
            throw std::runtime_error("Unknown DataType!");
    }
}

} // namespace mini_nccl