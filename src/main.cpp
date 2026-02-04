#include "mini_nccl.h"
#include "transport/RDMATransport.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./mini_nccl <rank> [server_ip]" << std::endl;
        return 1;
    }
    int rank = std::stoi(argv[1]);
    std::string root_ip = (argc > 2) ? argv[2] : "127.0.0.1";
    int size = 2; // 固定 2 个节点测试

    std::cout << "[Mini-NCCL] Rank " << rank << " starting..." << std::endl;

    // 1. 创建 Transport
    auto transport = std::make_shared<mini_nccl::RDMATransport>(rank, size, root_ip);
    transport->init(); // 握手

    // 2. 创建 Context
    auto ctx = std::make_shared<mini_nccl::Context>(rank, size, transport);

    // 3. 准备数据 (Pinned Memory)
    int count = 1024 * 1024; // 1M float
    size_t bytes = count * sizeof(float);
    float* data;
    cudaHostAlloc(&data, bytes, cudaHostAllocDefault);

    // 初始化：Rank 0 全是 1.0, Rank 1 全是 2.0
    for(int i=0; i<count; ++i) data[i] = (rank == 0) ? 1.0f : 2.0f;

    std::cout << "[Mini-NCCL] Starting All-Reduce..." << std::endl;

    // 4. 调用核心库
    mini_nccl::allreduce(data, count, ctx);

    // 5. 验证结果
    bool pass = true;
    for(int i=0; i<count; ++i) {
        if (data[i] != 3.0f) {
            pass = false;
            if (i < 10) std::cerr << "Mismatch at " << i << " expected 3.0 got " << data[i] << std::endl;
        }
    }

    if (pass) std::cout << "Result: [PASS] All values are 3.0!" << std::endl;
    else std::cout << "Result: [FAIL]" << std::endl;

    cudaFreeHost(data);
    return 0;
}