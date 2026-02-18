#pragma once
#include "mini_nccl.h" 
#include <atomic>
#include <vector>
#include <stdexcept>

namespace mini_nccl {

template<typename T>
class LockFreeQueue {
private:
    std::vector<T> buffer_;
    size_t capacity_;
    alignas(64) std::atomic<size_t> head_{0}; // Consumer index
    alignas(64) std::atomic<size_t> tail_{0}; // Producer index

public:
    explicit LockFreeQueue(size_t capacity) : capacity_(capacity) {
        // 环形缓冲区多留一个空位用于区分空/满
        buffer_.resize(capacity + 1);
    }

    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    // Producer 调用: 入队 (非阻塞)
    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % buffer_.size();

        // 检查是否已满
        // 优化: 在高性能场景下，我们假设队列容量足够大(4096)，满是小概率事件
        // 使用 unlikely 提示编译器将错误处理逻辑移出热路径
        if (unlikely(next_tail == head_.load(std::memory_order_acquire))) {
            return false; 
        }

        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release); 
        return true;
    }

    // Consumer 调用: 出队 (非阻塞)
    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        
        // 检查是否为空
        // 注意: 对于轮询消费者，空队列可能是常态，所以这里不加 unlikely
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; 
        }

        item = buffer_[current_head];
        head_.store((current_head + 1) % buffer_.size(), std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }
};

} // namespace mini_nccl