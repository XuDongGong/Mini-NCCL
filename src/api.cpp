#include "mini_nccl_api.h"
#include "mini_nccl.h"             
#include "transport/RDMATransport.h" 
#include <iostream>
#include <cstring>
#include <cuda_runtime.h> // for cudaPointerGetAttributes

using namespace mini_nccl;

extern "C" {

const char* ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess: return "no error";
        case ncclUnhandledCudaError: return "unhandled cuda error";
        case ncclSystemError: return "system error";
        case ncclInternalError: return "internal error";
        case ncclInvalidArgument: return "invalid argument";
        case ncclInvalidUsage: return "invalid usage";
        default: return "unknown error";
    }
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nRanks, int rank, const char* ip) {
    if (!comm || nRanks <= 0 || rank < 0 || rank >= nRanks) return ncclInvalidArgument;
    try {
        std::string root_ip = (ip) ? std::string(ip) : "127.0.0.1";
        auto transport = std::make_shared<RDMATransport>(rank, nRanks, root_ip);
        // 注意：这里只做基础的初始化，不应包含过重的资源分配，防止这里耗时太久
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
    try {
        Context* ctx = (Context*)comm;
        delete ctx;
        return ncclSuccess;
    } catch (...) {
        return ncclSystemError;
    }
}

// =============================================================
// 辅助函数：类型与参数校验
// =============================================================

size_t get_type_size(ncclDataType_t type) {
    switch(type) {
        case ncclFloat:  return 4;
        case ncclInt32:  return 4;
        case ncclDouble: return 8;
        // 显式列出不支持的类型，便于后续扩展
        case ncclInt8:
        case ncclUint8:
        case ncclUint32:
        case ncclInt64:
        case ncclUint64:
        case ncclFloat16:
        case ncclBfloat16:
        default:
            throw std::runtime_error("Unsupported DataType");
    }
}

DataType to_internal_dtype(ncclDataType_t t) {
    switch(t) {
        case ncclFloat:  return DataType::Float32;
        case ncclDouble: return DataType::Float64;
        case ncclInt32:  return DataType::Int32;
        default: throw std::runtime_error("Unsupported DataType conversion");
    }
}

RedOp to_internal_op(ncclRedOp_t op) {
    switch(op) {
        case ncclSum:  return RedOp::Sum;
        case ncclProd: return RedOp::Prod;
        case ncclMax:  return RedOp::Max;
        case ncclMin:  return RedOp::Min;
        default: throw std::runtime_error("Unsupported RedOp conversion");
    }
}

// 严格的内存检查：当前版本只支持 Pinned Host Memory
void check_pointer_attributes(const void* ptr, const char* name) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    
    // 如果无法获取属性（可能是普通 malloc 内存），但在 Unified Addressing 下通常能返回
    if (err != cudaSuccess) {
        // 清除错误状态
        cudaGetLastError();
        throw std::runtime_error(std::string(name) + " is not a valid CUDA pointer (did you use cudaHostAlloc?)");
    }

    // 检查是否为 Host 内存
    if (attr.type != cudaMemoryTypeHost) {
        // 允许 Managed Memory (Unified Memory)，但在当前 RDMA 实现下可能有性能问题，暂且放行
        if (attr.type == cudaMemoryTypeDevice) {
             throw std::runtime_error(std::string(name) + " is Device Memory. Current version ONLY supports Pinned Host Memory (cudaHostAlloc).");
        }
    }
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    // 1. 基础参数校验
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess; // 按照语义，空操作直接成功

    Context* ctx = (Context*)comm;

    try {
        // 2. 类型与大小校验
        size_t type_size = get_type_size(datatype);
        size_t total_bytes = count * type_size;

        // 3. 内存属性校验 (P0 级安全检查)
        check_pointer_attributes(sendbuff, "sendbuff");
        check_pointer_attributes(recvbuff, "recvbuff");

        // 4. In-place 处理
        // 如果是 Out-of-place (send != recv)，需要先拷贝到 recv
        // 注意：这里假设了后续 kernel 是 In-place 的 (a+=b -> a)
        if (sendbuff != recvbuff) {
            cudaMemcpyAsync(recvbuff, sendbuff, total_bytes, cudaMemcpyHostToHost, stream);
        }

        // 5. 创建 Alias Shared Ptr (不拥有所有权)
        std::shared_ptr<Context> ctx_ptr(ctx, [](Context*){}); 
        
        // 6. 调用内部实现
        mini_nccl::allreduce(recvbuff, count, to_internal_dtype(datatype), to_internal_op(op), ctx_ptr, stream);
        
        return ncclSuccess;

    } catch (const std::runtime_error& e) {
        std::cerr << "[Mini-NCCL] AllReduce Argument Error: " << e.what() << std::endl;
        return ncclInvalidArgument;
    } catch (const std::exception& e) {
        std::cerr << "[Mini-NCCL] AllReduce Internal Error: " << e.what() << std::endl;
        return ncclInternalError;
    } catch (...) {
        return ncclSystemError;
    }
}

} // extern "C"