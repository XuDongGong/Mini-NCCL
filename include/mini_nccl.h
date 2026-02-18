#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>

// 告诉编译器某条分支发生的概率极低 (unlikely) 或极高 (likely)
// 编译器会将 "likely" 的代码紧凑排列，减少指令缓存 (I-Cache) 未命中
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
    // 暴露 RKey 给单边通信使用
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

    // --- 单边通信与信号系统接口 ---
    
    // 获取本地 Flag 数组的 GPU 可访问指针
    virtual uint32_t* get_flags_ptr() = 0;

    // RDMA Write (单边写数据)
    // Step 2.1: 增加 signaled 参数，默认 true
    virtual Request* write(int rank, std::shared_ptr<MemoryRegion> local_mr, size_t offset, size_t length,
                           uint64_t remote_addr, uint32_t remote_rkey, bool signaled = true) = 0;

    // RDMA Write Signal (单边写信号)
    // Step 2.1: 增加 signaled 参数，默认 true
    virtual Request* write_signal(int rank, int flag_idx, uint32_t value, bool signaled = true) = 0;
};

class Context {
public:
    Context(int rank, int size, std::shared_ptr<Transport> transport)
        : rank_(rank), size_(size), transport_(transport) {}
    int rank() const { return rank_; }
    int size() const { return size_; }
    std::shared_ptr<Transport> transport() const { return transport_; }
    
    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) {
        return transport_->registerMemory(ptr, size);
    }
private:
    int rank_;
    int size_;
    std::shared_ptr<Transport> transport_;
};

// 内部实现入口
void allreduce(void* data, int count, DataType dtype, RedOp op, std::shared_ptr<Context> ctx, cudaStream_t stream);

} // namespace mini_nccl