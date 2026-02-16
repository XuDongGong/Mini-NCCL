#include "mini_nccl_api.h"
#include "mini_nccl.h"             
#include "transport/RDMATransport.h" 
#include "hera/hera_worker.h" // 引入 Hera
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
    if (!comm) return ncclInvalidArgument;
    try {
        int final_rank = rank;
        int final_size = nRanks;
        std::string final_root_ip = (ip) ? std::string(ip) : "127.0.0.1";

        // ========================================================
        // Hera-Core Integration
        // ========================================================
        if (rank == -1) {
            std::cout << "[Mini-NCCL] Hera Mode Activated. Connecting to Master at " << final_root_ip << "..." << std::endl;
            
            // 1. 创建 Hera Worker 并连接 Master
            // 注意: 这里我们把 'comm' 指针强转为 Context 之前，还没地方存 HeraWorker
            // 为了简单起见，我们在这里局部创建，获取信息后销毁 (Agent 暂时只做 Bootstrap)
            // 在 Sprint 3 后续，Context 应该持有 HeraWorker 以处理心跳
            
            hera::HeraWorker agent(final_root_ip, 9999);
            agent.ConnectAndRegister();
            
            final_rank = agent.rank();
            final_size = agent.size();
            final_root_ip = agent.root_ip(); // 使用 Master 告诉我们的 Root IP (Rank 0 IP)
            
            std::cout << "[Mini-NCCL] Auto-Assigned Rank: " << final_rank 
                      << " Size: " << final_size 
                      << " Root-IP: " << final_root_ip << std::endl;
        }
        // ========================================================

        if (final_rank < 0 || final_rank >= final_size) return ncclInvalidArgument;

        auto transport = std::make_shared<RDMATransport>(final_rank, final_size, final_root_ip);
        transport->init();
        
        auto context = new Context(final_rank, final_size, transport);
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

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (!comm || !rank) return ncclInvalidArgument;
    try {
        Context* ctx = (Context*)comm;
        *rank = ctx->rank();
        return ncclSuccess;
    } catch (...) {
        return ncclSystemError;
    }
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (!comm || !count) return ncclInvalidArgument;
    try {
        Context* ctx = (Context*)comm;
        *count = ctx->size();
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

void check_pointer_attributes(const void* ptr, const char* name) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) { cudaGetLastError(); return; }
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

        if (sendbuff != recvbuff) {
            cudaMemcpyAsync(recvbuff, sendbuff, total_bytes, cudaMemcpyDeviceToDevice, stream);
        }

        std::shared_ptr<Context> ctx_ptr(ctx, [](Context*){}); 
        mini_nccl::allreduce(recvbuff, count, to_internal_dtype(datatype), to_internal_op(op), ctx_ptr, stream);
        
        return ncclSuccess;
    } catch (const std::exception& e) {
        std::cerr << "[Mini-NCCL] AllReduce Internal Error: " << e.what() << std::endl;
        return ncclInternalError;
    } catch (...) {
        return ncclSystemError;
    }
}

} // extern "C"