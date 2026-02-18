#pragma once

#include "mini_nccl_api.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>

// 告诉编译器某条分支发生的概率极低 (unlikely) 或极高 (likely)
#if defined(__GNUC__) || defined(__clang__)
    #define likely(x)       __builtin_expect(!!(x), 1)
    #define unlikely(x)     __builtin_expect(!!(x), 0)
#else
    #define likely(x)       (x)
    #define unlikely(x)     (x)
#endif

namespace mini_nccl {

// 定义内部使用的枚举
enum class DataType {
    Float32 = 0,
    Float64 = 1,
    Int32   = 2
};

enum class RedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3
};

class MemoryRegion {
public:
    virtual ~MemoryRegion() = default;
    virtual void* ptr() const = 0;
    virtual size_t size() const = 0;
    virtual uint32_t rkey() const = 0;
};

class Request {
public:
    virtual ~Request() = default;
    virtual void wait() = 0;
    virtual bool isCompleted() const = 0;
    virtual void release() = 0;
};

class Transport {
public:
    virtual ~Transport() = default;
    virtual void init() = 0;
    
    virtual Request* isend(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) = 0;
    virtual Request* irecv(int rank, std::shared_ptr<MemoryRegion> mr, size_t offset, size_t length) = 0;
    virtual std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) = 0;

    virtual uint32_t* get_flags_ptr() = 0;
    virtual uint32_t* get_abort_flag_dev_ptr() = 0;
    virtual void abort() = 0;

    virtual Request* write(int rank, std::shared_ptr<MemoryRegion> local_mr, size_t offset, size_t length,
                           uint64_t remote_addr, uint32_t remote_rkey, bool signaled = true) = 0;

    virtual Request* write_signal(int rank, int flag_idx, uint32_t value, bool signaled = true) = 0;
    
    // 注意：这里为了保持 API 简单，我们暂时不在基类展开 DynamicInfo 的参数
    // 实际调用时会使用 dynamic_pointer_cast<RDMATransport>
};

class Context {
public:
    // 修改: 构造函数改为声明，实现在 .cu 文件中
    Context(int rank, int size, std::shared_ptr<Transport> transport);
    
    ~Context() {
        if (host_buffer_) cudaFreeHost(host_buffer_);
    }

    int rank() const { return rank_; }
    int size() const { return size_; }
    std::shared_ptr<Transport> transport() const { return transport_; }
    
    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) {
        return transport_->registerMemory(ptr, size);
    }

    // >>> 🚀 提升五: 内存复用机制接口 >>>
    void* get_scratch_buffer(int idx) {
        // 简单的双缓冲索引检查
        if (idx < 0 || idx > 1) return nullptr;
        if (!host_buffer_) return nullptr;
        return (char*)host_buffer_ + idx * max_slice_size_;
    }

    void allocate_scratch_buffer(size_t slice_size) {
        if (host_buffer_) return; 
        max_slice_size_ = slice_size; 
        // 关键点: 使用 cudaHostAllocMapped 
        // 1. 允许 GPU Kernel 直接访问 (Zero-Copy)
        // 2. 避免 "invalid argument" 或 Sticky Error
        cudaError_t err = cudaHostAlloc(&host_buffer_, max_slice_size_ * 2, cudaHostAllocMapped);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate context scratch buffer: " + std::string(cudaGetErrorString(err)));
        }
    }
    // <<< 提升五结束 <<<

private:
    int rank_;
    int size_;
    std::shared_ptr<Transport> transport_;
    
    // 内存池变量
    void* host_buffer_ = nullptr;
    size_t max_slice_size_ = 0;
};

void allreduce(void* data, int count, DataType dtype, RedOp op, std::shared_ptr<Context> ctx, cudaStream_t stream);

} // namespace mini_nccl