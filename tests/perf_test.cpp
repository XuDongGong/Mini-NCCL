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

double get_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rank> <n_ranks> [master_ip]" << std::endl;
        std::cerr << "       Pass rank = -1 to enable Hera Auto-Networking" << std::endl;
        return 1;
    }

    int rank_arg = atoi(argv[1]); // 用户输入的 Rank (可能是 -1)
    int nRanks = atoi(argv[2]);
    const char* ip = (argc > 3) ? argv[3] : "127.0.0.1";

    // 绑定设备：为了简单起见，WSL2 模拟环境下统一绑定 GPU 0
    CUDA_CHECK(cudaSetDevice(0)); 

    ncclComm_t comm;
    // 初始化通信域 (如果 rank_arg 是 -1，将触发 Hera 自动组网)
    NCCL_CHECK(ncclCommInitRank(&comm, nRanks, rank_arg, ip));

    // >>> 关键逻辑：如果是自动模式，反向查询我到底是谁 <<<
    int final_rank = rank_arg;
    if (rank_arg == -1) {
        NCCL_CHECK(ncclCommUserRank(comm, &final_rank));
        // 也可以查一下总数
        // int final_count; NCCL_CHECK(ncclCommCount(comm, &final_count));
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 测试梯度
    std::vector<size_t> sizes = {
        1 * 1024 * 1024UL,
        16 * 1024 * 1024UL,
        64 * 1024 * 1024UL
    };

    // 只有 Rank 0 打印表头
    if (final_rank == 0) {
        printf("\n=== Mini-NCCL Performance Benchmark (Rank %d) ===\n", final_rank);
        printf("%15s %15s %15s %15s\n", "Size(B)", "Time(us)", "AlgBW(GB/s)", "BusBW(GB/s)");
    }

    for (size_t size : sizes) {
        size_t count = size / sizeof(float);
        float* d_send;
        float* d_recv;

        CUDA_CHECK(cudaHostAlloc(&d_send, size, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&d_recv, size, cudaHostAllocDefault));

        // 预热
        for (int i = 0; i < 3; ++i) {
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 计时
        int iterations = 10;
        double start = get_us();
        for (int i = 0; i < iterations; ++i) {
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double end = get_us();

        double avg_time_us = (end - start) / iterations;
        double alg_bw = (double)size / avg_time_us / 1e3; 
        double factor = 2.0 * (nRanks - 1) / (double)nRanks;
        double bus_bw = alg_bw * factor;

        if (final_rank == 0) {
            printf("%15lu %15.2f %15.2f %15.2f\n", size, avg_time_us, alg_bw, bus_bw);
        }

        CUDA_CHECK(cudaFreeHost(d_send));
        CUDA_CHECK(cudaFreeHost(d_recv));
        usleep(5000);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));

    return 0;
}