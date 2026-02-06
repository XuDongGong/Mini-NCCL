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
    if (!comm || nRanks <= 0 || rank < 0 || rank >= nRanks) return ncclInvalidArgument;
    try {
        std::string root_ip = (ip) ? std::string(ip) : "127.0.0.1";
        auto transport = std::make_shared<RDMATransport>(rank, nRanks, root_ip);
        transport->init();
        auto context = new Context(rank, nRanks, transport);
        *comm = (ncclComm_t)context;
        return ncclSuccess;
    } catch (const std::exception& e) {
        std::cerr << "[Mini-NCCL] Init Failed: " << e.what() << std::endl;
        return ncclSystemError;
    }
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!comm) return ncclInvalidArgument;
    Context* ctx = (Context*)comm;
    delete ctx;
    return ncclSuccess;
}

// 辅助函数：转换枚举
DataType to_internal_dtype(ncclDataType_t t) {
    switch(t) {
        case ncclFloat: return DataType::Float32;
        // 如果 mini_nccl_api.h 里还没定义 ncclDouble/Int，这里需要扩展头文件或暂时映射
        // 假设我们在 api.h 只定义了 ncclFloat = 7
        // 为了演示，我们暂时只处理 float，其他报错或扩展
        default: return DataType::Float32; 
    }
}

RedOp to_internal_op(ncclRedOp_t op) {
    switch(op) {
        case ncclSum: return RedOp::Sum;
        // 同样，如果 api.h 没定义其他，暂时默认 Sum
        default: return RedOp::Sum;
    }
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    
    Context* ctx = (Context*)comm;

    // In-place check
    if (sendbuff != recvbuff) {
        size_t type_size = 4; // 简化：假设是 float
        if (datatype == ncclFloat) type_size = 4;
        cudaMemcpyAsync(recvbuff, sendbuff, count * type_size, cudaMemcpyDeviceToDevice, stream);
    }

    try {
        // 创建 Alias Shared Ptr (不拥有所有权)
        std::shared_ptr<Context> ctx_ptr(ctx, [](Context*){}); 
        
        // 调用内部泛型接口
        mini_nccl::allreduce(recvbuff, count, to_internal_dtype(datatype), to_internal_op(op), ctx_ptr, stream);
        
        return ncclSuccess;
    } catch (const std::exception& e) {
        std::cerr << "[Mini-NCCL] AllReduce Failed: " << e.what() << std::endl;
        return ncclInternalError;
    }
}
} // extern "C"