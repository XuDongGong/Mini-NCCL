#include "mini_nccl.h"
#include "transport/RDMATransport.h"
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

namespace mini_nccl {

const size_t SLICE_SIZE = 128 * 1024;
const int ELEMS_PER_SLICE = SLICE_SIZE / sizeof(float);

__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

void launch_cuda_kernel(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, n);
}

void allreduce(float* data, int count, std::shared_ptr<Context> ctx, cudaStream_t stream) {
    int rank = ctx->rank();
    int size = ctx->size();
    if (size == 1) return;

    int chunk_count = count / size;
    size_t chunk_bytes = chunk_count * sizeof(float);
    int recv_from = (rank - 1 + size) % size;
    int send_to   = (rank + 1) % size;

    auto mr_data = ctx->registerMemory(data, count * sizeof(float));

    float* buffers[2];
    std::shared_ptr<MemoryRegion> mr_buffers[2];
    for(int i=0; i<2; ++i) {
        checkCuda(cudaHostAlloc(&buffers[i], SLICE_SIZE, cudaHostAllocDefault), "Alloc Double Buffer");
        mr_buffers[i] = ctx->registerMemory(buffers[i], SLICE_SIZE);
    }

    // =============================================================
    // Phase 1: Scatter-Reduce (Object Pool Version)
    // =============================================================
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + size) % size;
        int recv_idx = (rank - i - 1 + size) % size;
        size_t block_send_offset = send_idx * chunk_bytes;
        int num_slices = (chunk_bytes + SLICE_SIZE - 1) / SLICE_SIZE;

        size_t slice_0_bytes = std::min(SLICE_SIZE, chunk_bytes);
        
        // 1. Pre-recv Slice 0
        Request* req_recv_next = ctx->transport()->irecv(recv_from, mr_buffers[0], 0, slice_0_bytes);

        for (int s = 0; s < num_slices; ++s) {
            size_t current_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - s * SLICE_SIZE);
            int current_slice_elems = current_slice_bytes / sizeof(float);
            
            int curr_buff_idx = s % 2;
            int next_buff_idx = (s + 1) % 2;

            Request* req_recv_future = nullptr;
            if (s < num_slices - 1) {
                size_t next_slice_bytes = std::min(SLICE_SIZE, chunk_bytes - (s + 1) * SLICE_SIZE);
                req_recv_future = ctx->transport()->irecv(recv_from, mr_buffers[next_buff_idx], 0, next_slice_bytes);
            }

            Request* req_send = ctx->transport()->isend(send_to, mr_data, block_send_offset + s * SLICE_SIZE, current_slice_bytes);

            // Wait current recv
            req_recv_next->wait();
            
            // --- 👇 关键点: 用完即还 ---
            req_recv_next->release(); 

            // Compute
            float* d_target = data + (recv_idx * chunk_count) + (s * ELEMS_PER_SLICE);
            launch_cuda_kernel(d_target, buffers[curr_buff_idx], d_target, current_slice_elems, stream);

            // Wait send
            req_send->wait();
            req_send->release(); // --- 👇 关键点: 用完即还 ---

            // Move pointer
            req_recv_next = req_recv_future;
        }
        checkCuda(cudaStreamSynchronize(stream), "Stream Sync");
    }

    // =============================================================
    // Phase 2: All-Gather (Object Pool Version)
    // =============================================================
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
            
            // --- 👇 释放资源 ---
            req_recv->release();
            req_send->release();
        }
    }

    for(int i=0; i<2; ++i) cudaFreeHost(buffers[i]);
}

} // namespace mini_nccl