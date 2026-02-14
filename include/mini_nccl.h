#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>

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
    // 注意：需要知道远程 buffer 的地址和 rkey (通常通过带外方式交换或预先计算)
    // 为了简化，我们假设上层知道 remote_addr
    virtual Request* write(int rank, std::shared_ptr<MemoryRegion> local_mr, size_t offset, size_t length,
                           uint64_t remote_addr, uint32_t remote_rkey) = 0;

    // RDMA Write Signal (单边写信号)
    virtual Request* write_signal(int rank, int flag_idx, uint32_t value) = 0;
};

class Context {
public:
    Context(int rank, int size, std::shared_ptr<Transport> transport)
        : rank_(rank), size_(size), transport_(transport) {}
    int rank() const { return rank_; }
    int size() const { return size_; }
    std::shared_ptr<Transport> transport() { return transport_; }
    std::shared_ptr<MemoryRegion> registerMemory(void* ptr, size_t size) {
        return transport_->registerMemory(ptr, size);
    }
private:
    int rank_;
    int size_;
    std::shared_ptr<Transport> transport_;
};

// 核心泛型接口
void allreduce(void* data, int count, DataType dtype, RedOp op, std::shared_ptr<Context> ctx, cudaStream_t stream);

} // namespace mini_nccl