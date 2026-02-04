#include "mini_nccl.h"
#include "transport/RDMATransport.h"
#include <iostream>
#include <cmath>

namespace mini_nccl {

// CUDA 核函数：向量加法
__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

// 核心算法：Ring All-Reduce
void allreduce(float* data, int count, std::shared_ptr<Context> ctx) {
    int rank = ctx->rank();
    int size = ctx->size();
    
    // 如果只有一个人，直接返回
    if (size == 1) return;

    // 1. 计算分块大小
    // 为了简单，我们假设 count 能被 size 整除
    // 实际工程中需要处理 padding
    int chunk_size = count / size;
    if (count % size != 0) {
        std::cerr << "[Warning] Data count is not divisible by rank size. Tail data ignored." << std::endl;
    }
    size_t chunk_bytes = chunk_size * sizeof(float);

    // 2. 准备邻居
    int recv_from = (rank - 1 + size) % size; // 左边
    int send_to   = (rank + 1) % size;        // 右边

    // 3. 注册内存 (整个 data 数组)
    // 注意：RDMA 需要注册整个缓冲区
    auto mr = ctx->registerMemory(data, count * sizeof(float));

    // 4. 准备 GPU 上的临时缓冲区 (用于接收左边发来的数据)
    // 真正的 NCCL 会用 double buffering，这里用简单的单缓冲
    // 此时我们需要一个 "Receiving Buffer"。
    // Ring 算法中，我们不能直接覆盖 data，因为我们还需要发给右边。
    // 但是 RDMA 必须收数据到注册内存。
    // 简单起见，我们在 data 后面（或者是另外分配的一块注册内存）作为 buffer。
    // 为了不破坏接口，我们假设 data 已经在 GPU 上了。
    // 等等！Soft-RoCE 不能直接访问 GPU 显存。
    // 我们之前的 Demo 01 是用 cudaHostAlloc (CPU Pinned Memory)。
    // 这里的 `data` 指针，用户传进来的如果是 GPU 指针，Soft-RoCE 会挂。
    
    // *** 关键假设 ***：
    // 目前版本，我们假设 `data` 是通过 cudaHostAlloc 分配的 CPU Pinned Memory。
    // 或者我们在此处分配临时 CPU Buffer 进行中转（性能会低，但是通用）。
    
    // 为了高性能验证，我们暂定：用户传进来的 data 必须是 cudaHostAlloc 的内存。
    // 这样 GPU 能算，RDMA 也能传。

    // 分配一个临时的接收 buffer (也是 Pinned Memory)
    float* recv_buffer;
    checkCuda(cudaHostAlloc(&recv_buffer, chunk_bytes, cudaHostAllocDefault), "Alloc recv buffer");
    auto mr_recv = ctx->registerMemory(recv_buffer, chunk_bytes);

    // =============================================================
    // Phase 1: Scatter-Reduce
    // =============================================================
    // 逻辑：N-1 轮。每轮发送 chunk[send_idx]，接收 chunk[recv_idx]，然后加到 data[recv_idx] 上。
    
    for (int i = 0; i < size - 1; ++i) {
        // 算出这一轮我要处理哪一块数据
        int send_idx = (rank - i + size) % size;
        int recv_idx = (rank - i - 1 + size) % size;

        size_t send_offset = send_idx * chunk_bytes;
        // 接收 offset 设置为 0 (收在临时 buffer 里)

        // 1. 启动接收 (先 Recv 再 Send 是防止死锁的好习惯，虽然异步无所谓)
        auto req_recv = ctx->transport()->irecv(recv_from, mr_recv, 0, chunk_bytes);
        
        // 2. 启动发送
        auto req_send = ctx->transport()->isend(send_to, mr, send_offset, chunk_bytes);

        // 3. 等待传输完成
        req_recv->wait();
        req_send->wait();

        // 4. 计算：data[recv_idx] += recv_buffer
        // 由于 data 和 recv_buffer 都是 CPU Pinned Memory，GPU 可以直接访问
        float* d_target = data + (recv_idx * chunk_size);
        float* d_source = recv_buffer;
        
        // 启动 GPU 核函数进行加法
        // 注意：虽然内存在 CPU 上，但因为是 Pinned，GPU 也可以访问 (Zero-Copy)
        // 实际上为了速度，应该 cudaMemcpy 到 Device 算完再拷回，或者 data 本身就在 Device (需 GPUDirect)。
        // 这里为了兼容 Demo 01 环境，我们用最简单的方式：CPU 上算！
        // 哎呀，如果用 GPU 核函数，还得配置 stream。
        // Demo 01 里我们是 cudaMemcpy 进 GPU 算的。
        // 这里简化：直接用 CPU 算 (模拟 GPU)，因为数据本身就在 Host Memory。
        // 这样代码更简洁，专注验证 RDMA Ring 逻辑。
        
        for(int k=0; k<chunk_size; ++k) {
            d_target[k] += d_source[k];
        }
    }

    // =============================================================
    // Phase 2: All-Gather
    // =============================================================
    // 逻辑：N-1 轮。每轮直接把我已经算好的数据发给右边，接收左边的数据填空。
    
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + 1 + size) % size;
        int recv_idx = (rank - i + size) % size;
        
        size_t send_offset = send_idx * chunk_bytes;
        size_t recv_offset = recv_idx * chunk_bytes;

        // 1. 接收 (这次直接收进目标位置，不需要累加了)
        // 这里的 mr 是 data 的 MR，offset 是 recv_idx
        auto req_recv = ctx->transport()->irecv(recv_from, mr, recv_offset, chunk_bytes);
        
        // 2. 发送
        auto req_send = ctx->transport()->isend(send_to, mr, send_offset, chunk_bytes);

        // 3. 等待
        req_recv->wait();
        req_send->wait();
    }

    // 清理
    cudaFreeHost(recv_buffer);
}

} // namespace mini_nccl