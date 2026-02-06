#ifndef MINI_NCCL_API_H_
#define MINI_NCCL_API_H_

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================
// 1. 类型定义
// =============================================================

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4
} ncclResult_t;

typedef struct ncclComm* ncclComm_t;

// 完整的 NCCL 数据类型定义
typedef enum { 
    ncclInt8       = 0, 
    ncclUint8      = 1, 
    ncclInt32      = 2, 
    ncclUint32     = 3, 
    ncclInt64      = 4, 
    ncclUint64     = 5, 
    ncclFloat16    = 6, 
    ncclFloat      = 7, 
    ncclDouble     = 8, 
    ncclBfloat16   = 9 
} ncclDataType_t;

// 完整的 NCCL 操作定义
typedef enum { 
    ncclSum        = 0, 
    ncclProd       = 1, 
    ncclMax        = 2, 
    ncclMin        = 3, 
    ncclAvg        = 4 
} ncclRedOp_t;

// =============================================================
// 2. 核心接口
// =============================================================

const char* ncclGetErrorString(ncclResult_t result);

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nRanks, int rank, const char* ip);

ncclResult_t ncclCommDestroy(ncclComm_t comm);

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MINI_NCCL_API_H_