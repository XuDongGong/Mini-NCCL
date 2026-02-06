/**
 * Mini-NCCL: A lightweight collective communication library.
 * Designed for learning RDMA and Ring-AllReduce algorithms.
 * Architecture Reference: PyTorch Gloo
 * Author: gongxudong
 * email: markxdgong@outlook.com
 * Data: [2026/02/04]
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>

namespace mini_nccl {

// =============================================================
// 1. 内存抽象 (Memory Abstraction)
// =============================================================

/**
 * MemoryRegion: 封装 RDMA 的内存注册 (Memory Registration).
 * 在 RDMA 中，发送/接收的数据必须先注册到网卡，获取 lkey/rkey。
 * 普通 TCP 不需要这个，但为了统一接口，我们需要它。
 */
class MemoryRegion {
public:
    virtual ~MemoryRegion() = default;
    
    // 获取数据的原始指针
    virtual void* ptr() const = 0;
    
    // 获取数据大小
    virtual size_t size() const = 0;
};

// =============================================================
// 2. 传输层抽象 (Transport Abstraction)
// =============================================================

// 前置声明
class Context;

/**
 * Request: 异步操作的句柄
 */
class Request {
public:
    virtual ~Request() = default;
    virtual void wait() = 0;
    virtual bool isCompleted() const = 0;
    
    // 归还对象到资源池
    // 类似于 delete this，但实际上只是回收到池子里
    virtual void release() = 0;
};

/**
 * Transport: 负责底层的点对点通信
 * 抽象基类，需要实现具体的 RDMATransport。
 */
class Transport {
public:
    virtual ~Transport() = default;

    // 初始化传输层 (例如建立 QP 连接，或者 TCP 握手)
    virtual void init() = 0;

    // 返回值从 std::shared_ptr<Request> 改为 Request*
    // 含义：Caller 借用这个对象，用完必须调用 req->release() 归还
    virtual Request* isend(int rank, 
                           std::shared_ptr<MemoryRegion> mr, 
                           size_t offset, 
                           size_t length) = 0;

    virtual Request* irecv(int rank, 
                           std::shared_ptr<MemoryRegion> mr, 
                           size_t offset, 
                           size_t length) = 0;
    
    virtual std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) = 0;
};

// =============================================================
// 3. 上下文 (Context)
// =============================================================

/**
 * Context: 全局视图
 */
class Context {
public:
    Context(int rank, int size, std::shared_ptr<Transport> transport)
        : rank_(rank), size_(size), transport_(transport) {}

    // 基础信息
    int rank() const { return rank_; } // 当前节点的 ID (0, 1, 2...)
    int size() const { return size_; } // 总节点数

    // 获取传输层实例
    std::shared_ptr<Transport> transport() { return transport_; }

    // 简化的注册内存接口
    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) {
        return transport_->registerMemory(ptr, size);
    }

private:
    int rank_;
    int size_;
    std::shared_ptr<Transport> transport_;
};

// =============================================================
// 4. 算法接口 (Collectives)
// =============================================================

/**
 * 核心算法入口: AllReduce
 * data: 数据指针
 * count: 元素个数，不是字节数
 * ctx: 上下文
 */
void allreduce(float* data, int count, std::shared_ptr<Context> ctx, cudaStream_t stream);

} // namespace mini_nccl