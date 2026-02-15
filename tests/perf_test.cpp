#include "mini_nccl_api.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <cstring>
#include <ctime>

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if(e != cudaSuccess) { \
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if(r != ncclSuccess) { \
        printf("Failed: NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 获取高精度时间 (微秒)
double get_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rank> <n_ranks> [root_ip]" << std::endl;
        return 1;
    }

    int rank = atoi(argv[1]);
    int nRanks = atoi(argv[2]);
    const char* ip = (argc > 3) ? argv[3] : "127.0.0.1";

    // 绑定设备：Rank 0 -> GPU 0, Rank 1 -> GPU 0 (WSL通常只有一个GPU，模拟多卡)
    // 如果你有真多卡，这里应该是 cudaSetDevice(rank)
    CUDA_CHECK(cudaSetDevice(0)); 

    ncclComm_t comm;
    // 初始化通信域
    NCCL_CHECK(ncclCommInitRank(&comm, nRanks, rank, ip));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 测试梯度：1MB, 4MB, 16MB, 64MB, 128MB, 256MB
    std::vector<size_t> sizes = {
        1 * 1024 * 1024UL,
        4 * 1024 * 1024UL,
        16 * 1024 * 1024UL,
        64 * 1024 * 1024UL,
        128 * 1024 * 1024UL,
        256 * 1024 * 1024UL
    };

    if (rank == 0) {
        printf("\n=== Mini-NCCL Performance Benchmark ===\n");
        printf("%15s %15s %15s %15s\n", "Size(B)", "Time(us)", "AlgBW(GB/s)", "BusBW(GB/s)");
    }

    for (size_t size : sizes) {
        size_t count = size / sizeof(float);
        float* d_send;
        float* d_recv;

        // Phase 1 契约：必须使用 Pinned Host Memory
        CUDA_CHECK(cudaHostAlloc(&d_send, size, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&d_recv, size, cudaHostAllocDefault));

        // 预热 (Warmup) - 5次
        for (int i = 0; i < 5; ++i) {
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 计时 (Benchmark) - 20次
        int iterations = 20;
        double start = get_us();
        for (int i = 0; i < iterations; ++i) {
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double end = get_us();

        double avg_time_us = (end - start) / iterations;
        
        // 算法带宽 = 数据量 / 时间
        double alg_bw = (double)size / avg_time_us / 1e3; 
        
        // 总线带宽 (BusBW) = 算法带宽 * 2 * (N-1) / N
        // 对于 Ring 算法，这是衡量硬件利用率的标准指标
        double factor = 2.0 * (nRanks - 1) / (double)nRanks;
        double bus_bw = alg_bw * factor;

        if (rank == 0) {
            printf("%15lu %15.2f %15.2f %15.2f\n", size, avg_time_us, alg_bw, bus_bw);
        }

        CUDA_CHECK(cudaFreeHost(d_send));
        CUDA_CHECK(cudaFreeHost(d_recv));
        
        // 稍微休息一下，防止热节流
        usleep(10000);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));

    return 0;
}