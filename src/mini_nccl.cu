#include "mini_nccl.h"
#include "transport/RDMATransport.h"
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat> // for FLT_MAX etc.

namespace mini_nccl {

const size_t SLICE_SIZE = 128 * 1024; // 128KB Slice

// =============================================================
// 1. CUDA 算子模板 (Functors)
// =============================================================
template<typename T>
struct OpSum {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct OpProd {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};

template<typename T>
struct OpMax {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return (a > b) ? a : b; }
};

template<typename T>
struct OpMin {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return (a < b) ? a : b; }
};

// =============================================================
// 2. 泛型 CUDA Kernel
// =============================================================
template<typename T, typename Op>
__global__ void elementwise_reduce_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, int n, Op op) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = op(a[i], b[i]);
    }
}

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

// =============================================================
// 3. 泛型 All-Reduce 实现 (Internal Template)
// =============================================================
template<typename T, typename Op>
void allreduce_impl(T* data, int count, Op op, std::shared_ptr<Context> ctx, cudaStream_t stream) {
    int rank = ctx->rank();
    int size = ctx->size();
    if (size == 1) return;

    size_t type_size = sizeof(T);
    int chunk_count = count / size;
    size_t chunk_bytes = chunk_count * type_size;
    int recv_from = (rank - 1 + size) % size;
    int send_to   = (rank + 1) % size;

    // Slice 计算要基于 bytes 换算出 elements
    const int ELEMS_PER_SLICE = SLICE_SIZE / type_size;

    auto mr_data = ctx->registerMemory(data, count * type_size);

    // 双缓冲
    T* buffers[2];
    std::shared_ptr<MemoryRegion> mr_buffers[2];
    for(int i=0; i<2; ++i) {
        checkCuda(cudaHostAlloc(&buffers[i], SLICE_SIZE, cudaHostAllocDefault), "Alloc Double Buffer");
        mr_buffers[i] = ctx->registerMemory(buffers[i], SLICE_SIZE);
    }

    // --- Phase 1: Scatter-Reduce ---
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + size) % size;
        int recv_idx = (rank - i - 1 + size) % size;
        size_t block_send_offset = send_idx * chunk_bytes;
        
        int num_slices = (chunk_bytes + SLICE_SIZE - 1) / SLICE_SIZE;
        size_t slice_0_bytes = std::min(SLICE_SIZE, chunk_bytes);
        
        Request* req_recv_next = ctx->transport()->irecv(recv_from, mr_buffers[0], 0, slice_0_bytes);

        for (int s = 0; s < num_slices; ++s) {
            size_t current_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - s * SLICE_SIZE);
            int current_slice_elems = current_slice_bytes / type_size;
            
            int curr_buff_idx = s % 2;
            int next_buff_idx = (s + 1) % 2;

            Request* req_recv_future = nullptr;
            if (s < num_slices - 1) {
                size_t next_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - (s + 1) * SLICE_SIZE);
                req_recv_future = ctx->transport()->irecv(recv_from, mr_buffers[next_buff_idx], 0, next_slice_bytes);
            }

            Request* req_send = ctx->transport()->isend(send_to, mr_data, block_send_offset + s * SLICE_SIZE, current_slice_bytes);

            req_recv_next->wait();
            req_recv_next->release();

            // Launch Generic Kernel
            T* d_target = data + (recv_idx * chunk_count) + (s * ELEMS_PER_SLICE);
            int threads = 256;
            int blocks = (current_slice_elems + threads - 1) / threads;
            elementwise_reduce_kernel<<<blocks, threads, 0, stream>>>(d_target, buffers[curr_buff_idx], d_target, current_slice_elems, op);

            req_send->wait();
            req_send->release();

            req_recv_next = req_recv_future;
        }
        checkCuda(cudaStreamSynchronize(stream), "Stream Sync");
    }

    // --- Phase 2: All-Gather ---
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + 1 + size) % size;
        int recv_idx = (rank - i + size) % size;
        size_t block_send_offset = send_idx * chunk_bytes;
        size_t block_recv_offset = recv_idx * chunk_bytes;
        int num_slices = (chunk_bytes + SLICE_SIZE - 1) / SLICE_SIZE;

        for (int s = 0; s < num_slices; ++s) {
            size_t current_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - s * SLICE_SIZE);
            
            Request* req_recv = ctx->transport()->irecv(recv_from, mr_data, block_recv_offset + s * SLICE_SIZE, current_slice_bytes);
            Request* req_send = ctx->transport()->isend(send_to, mr_data, block_send_offset + s * SLICE_SIZE, current_slice_bytes);
            
            req_recv->wait();
            req_send->wait();
            req_recv->release();
            req_send->release();
        }
    }

    for(int i=0; i<2; ++i) cudaFreeHost(buffers[i]);
}

// =============================================================
// 4. 调度分发 (Dispatch Logic)
// =============================================================
// 这里的宏展示了如何处理 "Double Dispatch" (类型 x 操作)

#define DISPATCH_OP(TYPE_T, TYPE_ENUM, CTX, STREAM) \
    switch(op) { \
        case RedOp::Sum:  allreduce_impl(static_cast<TYPE_T*>(data), count, OpSum<TYPE_T>(), CTX, STREAM); break; \
        case RedOp::Prod: allreduce_impl(static_cast<TYPE_T*>(data), count, OpProd<TYPE_T>(), CTX, STREAM); break; \
        case RedOp::Max:  allreduce_impl(static_cast<TYPE_T*>(data), count, OpMax<TYPE_T>(), CTX, STREAM); break; \
        case RedOp::Min:  allreduce_impl(static_cast<TYPE_T*>(data), count, OpMin<TYPE_T>(), CTX, STREAM); break; \
    }

void allreduce(void* data, int count, DataType dtype, RedOp op, std::shared_ptr<Context> ctx, cudaStream_t stream) {
    switch(dtype) {
        case DataType::Float32:
            DISPATCH_OP(float, dtype, ctx, stream);
            break;
        case DataType::Float64:
            DISPATCH_OP(double, dtype, ctx, stream);
            break;
        case DataType::Int32:
            DISPATCH_OP(int, dtype, ctx, stream);
            break;
        default:
            std::cerr << "Unknown DataType!" << std::endl;
            exit(1);
    }
}

} // namespace mini_nccl