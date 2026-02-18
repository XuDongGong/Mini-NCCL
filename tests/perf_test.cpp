#include "mini_nccl_api.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <cstring>
#include <ctime>
#include <immintrin.h> //AVX2 指令集头文件

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

    int rank_arg = atoi(argv[1]); 
    int nRanks = atoi(argv[2]);
    const char* ip = (argc > 3) ? argv[3] : "127.0.0.1";

    // 简单绑定 GPU 0
    CUDA_CHECK(cudaSetDevice(0)); 

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nRanks, rank_arg, ip));

    // 如果是自动组网，查询真实 Rank
    int final_rank = rank_arg;
    if (rank_arg == -1) {
        NCCL_CHECK(ncclCommUserRank(comm, &final_rank));
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<size_t> sizes = {
        1 * 1024 * 1024UL,
        16 * 1024 * 1024UL,
        64 * 1024 * 1024UL,
        128 * 1024 * 1024UL
    };

    if (final_rank == 0) {
        printf("\n=== Mini-NCCL Performance Benchmark (Rank %d) ===\n", final_rank);
        printf("%15s %15s %15s %15s\n", "Size(B)", "Time(us)", "AlgBW(GB/s)", "BusBW(GB/s)");
    }

    for (size_t size : sizes) {
        size_t count = size / sizeof(float);
        float* d_send;
        float* d_recv;

        // 使用 Pinned Memory (通常是 4KB 对齐，满足 AVX2 的 32 字节对齐要求)
        CUDA_CHECK(cudaHostAlloc(&d_send, size, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&d_recv, size, cudaHostAllocDefault));

        // >>> 1. 数据初始化 <<<
        // 发送方全部填 1.0
        std::fill(d_send, d_send + count, 1.0f);
        // 接收方清零
        memset(d_recv, 0, size);

        // 预热
        for (int i = 0; i < 5; ++i) {
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 计时
        int iterations = 20;
        double start = get_us();
        for (int i = 0; i < iterations; ++i) {
            NCCL_CHECK(ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double end = get_us();

        // >>> 2. SIMD (AVX2) 极速校验 <<<
        // 预期结果：每个 Rank 都发了 1.0，AllReduce Sum 后应该是 nRanks * 1.0
        float expected_val = (float)nRanks; 
        bool pass = true;

        // 创建 AVX2 向量 [expected, expected, ..., expected]
        __m256 target_vec = _mm256_set1_ps(expected_val);
        
        int i = 0;
        // 每次处理 8 个 float (256 bits)
        // cudaHostAlloc 保证了内存对齐，所以可以用 _mm256_load_ps (Aligned Load)
        for (; i <= count - 8; i += 8) {
            __m256 loaded_data = _mm256_load_ps(&d_recv[i]);
            
            // 并行比较：不相等则对应的掩码位为 1
            // _CMP_NEQ_OQ: Not Equal (Ordered, Non-signaling)
            __m256 cmp_res = _mm256_cmp_ps(loaded_data, target_vec, _CMP_NEQ_OQ);
            
            // 将 256 位掩码压缩为 8 位整数
            int mask = _mm256_movemask_ps(cmp_res);
            
            if (mask != 0) {
                pass = false;
                // 找到具体的错误索引方便调试
                for(int k=0; k<8; ++k) {
                    if ((mask >> k) & 1) {
                        printf("[SIMD Check] Mismatch at index %d: expected %.1f, got %.1f\n", 
                                i+k, expected_val, d_recv[i+k]);
                        break; 
                    }
                }
                break; // 发现错误立即停止
            }
        }
        
        // 处理尾部剩余元素 (无法凑齐 8 个的部分)
        for (; i < count; ++i) {
            if (std::abs(d_recv[i] - expected_val) > 1e-5) {
                pass = false;
                printf("[Tail Check] Mismatch at index %d: expected %.1f, got %.1f\n", i, expected_val, d_recv[i]);
                break;
            }
        }

        if (!pass) {
            printf("[Rank %d] Verification FAILED for size %lu\n", final_rank, size);
            // 这里我们只打印错误，不退出，以便查看后续测试
        }

        double avg_time_us = (end - start) / iterations;
        double alg_bw = (double)size / avg_time_us / 1e3; 
        double factor = 2.0 * (nRanks - 1) / (double)nRanks;
        double bus_bw = alg_bw * factor;

        if (final_rank == 0) {
            printf("%15lu %15.2f %15.2f %15.2f %s\n", 
                size, avg_time_us, alg_bw, bus_bw, pass ? "" : "(FAIL)");
        }

        CUDA_CHECK(cudaFreeHost(d_send));
        CUDA_CHECK(cudaFreeHost(d_recv));
        usleep(10000);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));

    return 0;
}