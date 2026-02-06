#include "mini_nccl_api.h"
#include "mini_nccl.h"             // 引入我们的内部 C++ 定义
#include "transport/RDMATransport.h" // 引入 RDMA 实现
#include <iostream>
#include <cstring>

// 这里的 ncclComm 实际上就是我们在 mini_nccl.h 里定义的 Context
// 但为了 C 接口的兼容性，我们在这里做一个强制转换的桥接
// 我们需要确保 mini_nccl::Context 能被保存为 ncclComm_t

using namespace mini_nccl;

extern "C" {

const char* ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess: return "no error";
        case ncclUnhandledCudaError: return "unhandled cuda error";
        case ncclSystemError: return "system error";
        case ncclInternalError: return "internal error";
        case ncclInvalidArgument: return "invalid argument";
        default: return "unknown error";
    }
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nRanks, int rank, const char* ip) {
    if (!comm || nRanks <= 0 || rank < 0 || rank >= nRanks) {
        return ncclInvalidArgument;
    }

    try {
        // 1. 创建 Transport (C++ 对象)
        // 这里暂时硬编码 IP，或者使用传入的 ip 参数作为 root_ip
        std::string root_ip = (ip) ? std::string(ip) : "127.0.0.1";
        auto transport = std::make_shared<RDMATransport>(rank, nRanks, root_ip);
        
        // 2. 执行握手
        transport->init();

        // 3. 创建 Context (C++ 对象)
        auto context = new Context(rank, nRanks, transport);

        // 4. 将 C++ 对象指针强转为 C 句柄返回
        *comm = (ncclComm_t)context;

        return ncclSuccess;
    } catch (const std::exception& e) {
        std::cerr << "[Mini-NCCL] Init Failed: " << e.what() << std::endl;
        return ncclSystemError;
    }
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!comm) return ncclInvalidArgument;
    
    // 将 C 句柄转回 C++ 指针并 delete
    Context* ctx = (Context*)comm;
    delete ctx;
    
    return ncclSuccess;
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    
    // 暂时只支持 float sum
    if (datatype != ncclFloat || op != ncclSum) {
        std::cerr << "[Mini-NCCL] Only float sum is supported currently." << std::endl;
        return ncclInvalidArgument;
    }

    Context* ctx = (Context*)comm;

    // 如果 sendbuff 和 recvbuff 不一样，需要先拷贝
    // NCCL 标准允许 inplace (send==recv) 和 out-of-place
    // 我们的 mini_nccl::allreduce 目前是 inplace 的 (直接改数据)
    // 修正：我们的 allreduce 实现其实是在 data 上原地操作。
    // 如果用户要把 A 加到 B，我们需要先把 A 拷到 B，然后对 B 做 inplace。
    
    // 这里为了对接我们之前的 mini_nccl.cu 实现 (void allreduce(float* data...))
    // 我们做个简单的假设：目前只支持 In-Place 操作 (sendbuff == recvbuff)
    if (sendbuff != recvbuff) {
        // 真正的 NCCL 会处理这个，我们这里先偷懒，拷贝一下
        cudaMemcpyAsync(recvbuff, sendbuff, count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // 调用 C++ 核心逻辑
    // 注意：我们将 sendbuff 强转为 float*，因为之前的实现是 float*
    // 这里的 recvbuff 是我们要操作的目标
    try {
        // 由于我们之前的 allreduce 接收的是 shared_ptr<Context>，而这里我们持有的是裸指针
        // 这有点尴尬。为了不改动 mini_nccl.cu 太多，我们可以创建一个临时的 shared_ptr (不拥有所有权)
        // 或者修改 mini_nccl.cu 接受裸指针。
        // **最佳方案**：修改 api.cpp 适配 C++ 接口。
        
        // 创建一个不执行 delete 的 shared_ptr (Aliasing constructor)
        std::shared_ptr<Context> ctx_ptr(ctx, [](Context*){}); 
        
        mini_nccl::allreduce((float*)recvbuff, count, ctx_ptr, stream);
        
        return ncclSuccess;
    } catch (const std::exception& e) {
        std::cerr << "[Mini-NCCL] AllReduce Failed: " << e.what() << std::endl;
        return ncclInternalError;
    }
}

} // extern "C"