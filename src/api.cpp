#include "mini_nccl_api.h"
#include "mini_nccl.h"             
#include "transport/RDMATransport.h" 
#include <iostream>
#include <cstring>
#include <cuda_runtime.h> 

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

size_t get_type_size(ncclDataType_t type) {
    switch(type) {
        case ncclFloat:  return 4;
        case ncclInt32:  return 4;
        case ncclDouble: return 8;
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

// 升级版指针检查：支持 GPUDirect (Device Pointer)
void check_pointer_attributes(const void* ptr, const char* name) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    
    if (err != cudaSuccess) {
        // 允许普通 Host Memory (通过 OS Paging)
        cudaGetLastError(); 
        return; 
    }

    // Phase 5: 允许 Device Memory (开启 GPUDirect 路径)
    if (attr.type == cudaMemoryTypeDevice) {
        // Pass. transport layer will handle registration
    }
    else if (attr.type == cudaMemoryTypeHost) {
        // Pass. Pinned memory
    }
    // Managed memory 依然暂不支持，可能导致不可预期的性能问题
    else if (attr.type == cudaMemoryTypeManaged) {
         // Warn?
    }
}

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    if (count == 0) return ncclSuccess; 

    Context* ctx = (Context*)comm;

    try {
        size_t type_size = get_type_size(datatype);
        size_t total_bytes = count * type_size;

        check_pointer_attributes(sendbuff, "sendbuff");
        check_pointer_attributes(recvbuff, "recvbuff");

        if (sendbuff != recvbuff) {
            cudaMemcpyAsync(recvbuff, sendbuff, total_bytes, cudaMemcpyDeviceToDevice, stream);
        }

        std::shared_ptr<Context> ctx_ptr(ctx, [](Context*){}); 
        
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