#pragma once
#include <atomic>
#include <vector>
#include <stdexcept>

namespace mini_nccl {

template<typename T>
class LockFreeQueue {
private:
    std::vector<T> buffer_;
    size_t capacity_;
    // alignas(64) 防止伪共享 (False Sharing)，让 head 和 tail 处于不同的缓存行
    alignas(64) std::atomic<size_t> head_{0}; // Consumer index
    alignas(64) std::atomic<size_t> tail_{0}; // Producer index

public:
    explicit LockFreeQueue(size_t capacity) : capacity_(capacity) {
        // 环形缓冲区多留一个空位用于区分空/满
        buffer_.resize(capacity + 1);
    }

    // 禁止拷贝和赋值
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    // Producer 调用: 入队 (非阻塞)
    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % buffer_.size();

        // 检查是否已满 (Consumer 还没追上来)
        // 使用 acquire 确保读取到最新的 head
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; 
        }

        buffer_[current_tail] = item;
        // 发布数据: store release 保证 buffer 写入在 tail 更新之前完成
        tail_.store(next_tail, std::memory_order_release); 
        return true;
    }

    // Consumer 调用: 出队 (非阻塞)
    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        
        // 检查是否为空 (Producer 还没写入)
        // 使用 acquire 确保读取到最新的 tail
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; 
        }

        item = buffer_[current_head];
        // 发布状态: store release 告诉 Producer 这个槽位空出来了
        head_.store((current_head + 1) % buffer_.size(), std::memory_order_release);
        return true;
    }

    // 检查是否为空 (非强一致，仅供参考)
    bool empty() const {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }
};

} // namespace mini_nccl