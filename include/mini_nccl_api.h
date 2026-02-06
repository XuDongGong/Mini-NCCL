#ifndef MINI_NCCL_API_H_
#define MINI_NCCL_API_H_

#include <stddef.h>
#include <cuda_runtime.h>

// 启用 C 链接，防止 C++ 名字修饰 (Name Mangling)
#ifdef __cplusplus
extern "C" {
#endif

// =============================================================
// 1. 类型定义 (Opaque Types)
// =============================================================

// 错误码
typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4
} ncclResult_t;

// 通信器句柄 (不透明指针)
// 用户拿到的只是一个指针，不知道里面包含了 Transport/Socket 等复杂对象
typedef struct ncclComm* ncclComm_t;

// 数据类型 (目前我们只实现了 float)
typedef enum { 
    ncclFloat = 7 
} ncclDataType_t;

// 归约操作 (目前我们只实现了 Sum)
typedef enum { 
    ncclSum = 0 
} ncclRedOp_t;

// =============================================================
// 2. 核心接口 (Public API)
// =============================================================

// 获取错误信息
const char* ncclGetErrorString(ncclResult_t result);

// 初始化通信器
// nRanks: 总进程数
// rank: 当前进程 ID
// comm: 输出参数，返回创建好的句柄
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nRanks, int rank, const char* ip);

// 销毁通信器
ncclResult_t ncclCommDestroy(ncclComm_t comm);

// 核心计算: All-Reduce
// sendbuff: 发送缓冲区 (GPU 指针)
// recvbuff: 接收缓冲区 (GPU 指针)
// count: 元素数量
// datatype: 数据类型
// op: 归约操作类型
// comm: 通信器句柄
// stream: CUDA 流
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MINI_NCCL_API_H_