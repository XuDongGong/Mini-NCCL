// 注意：只引用公开的 C API 头文件
#include "mini_nccl_api.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <string>

// 宏定义检查错误
#define NCCL_CHECK(cmd) do {                  \
    ncclResult_t r = cmd;                     \
    if (r != ncclSuccess) {                   \
        std::cerr << "NCCL Error: " << ncclGetErrorString(r) << std::endl; \
        exit(1);                              \
    }                                         \
} while(0)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./app <rank> [server_ip]" << std::endl;
        return 1;
    }
    int rank = std::stoi(argv[1]);
    const char* ip = (argc > 2) ? argv[2] : "127.0.0.1";
    int nRanks = 2;

    std::cout << "[App] Rank " << rank << " starting..." << std::endl;

    // 1. 初始化 NCCL (标准 API)
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nRanks, rank, ip));

    // 2. 准备数据
    int count = 1024 * 1024;
    size_t bytes = count * sizeof(float);
    float* data;
    cudaHostAlloc(&data, bytes, cudaHostAllocDefault); // 使用 Pinned Memory
    
    // Init: Rank 0 -> 1.0, Rank 1 -> 2.0
    for(int i=0; i<count; ++i) data[i] = (rank == 0) ? 1.0f : 2.0f;

    // 3. 准备 Stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "[App] Calling ncclAllReduce..." << std::endl;

    // 4. 执行 AllReduce (标准 API)
    // In-place 操作: sendbuff == recvbuff
    NCCL_CHECK(ncclAllReduce(data, data, count, ncclFloat, ncclSum, comm, stream));

    // 5. 同步
    cudaStreamSynchronize(stream);

    // 6. 验证结果
    bool pass = true;
    for(int i=0; i<count; ++i) {
        if (data[i] != 3.0f) {
            pass = false;
            if (i < 5) std::cerr << "Mismatch at " << i << " expected 3.0 got " << data[i] << std::endl;
        }
    }

    if (pass) std::cout << "Result: [PASS] All values are 3.0!" << std::endl;
    else std::cout << "Result: [FAIL]" << std::endl;

    // 7. 清理
    NCCL_CHECK(ncclCommDestroy(comm));
    cudaStreamDestroy(stream);
    cudaFreeHost(data);

    return 0;
}