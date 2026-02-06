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