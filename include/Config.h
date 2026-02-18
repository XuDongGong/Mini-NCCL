#pragma once
#include <cstdlib>
#include <iostream>
#include <string>
#include <algorithm>

namespace mini_nccl {

class Config {
public:
    // 核心参数
    size_t slice_size;      // 切片大小 (默认 128KB)
    int window_size;        // 流控窗口大小 (默认 64)
    int signal_batch;       // 信号批次 (默认 16)
    bool gdr_enable;        // 是否开启 GPU Direct RDMA (预留)

    static Config& getInstance() {
        static Config instance;
        return instance;
    }

private:
    Config() {
        load_from_env();
    }

    void load_from_env() {
        // 1. Slice Size
        if (const char* env_p = std::getenv("MINI_NCCL_SLICE_SIZE")) {
            slice_size = std::stoul(env_p);
        } else {
            slice_size = 128 * 1024; // 默认 128KB
        }

        // 2. Window Size
        if (const char* env_p = std::getenv("MINI_NCCL_WINDOW_SIZE")) {
            window_size = std::stoi(env_p);
        } else {
            window_size = 64;
        }

        // 3. Signal Batch
        if (const char* env_p = std::getenv("MINI_NCCL_SIGNAL_BATCH")) {
            signal_batch = std::stoi(env_p);
        } else {
            signal_batch = 16;
        }

        // 简单校验
        if (slice_size == 0) slice_size = 1024;
        if (window_size <= 0) window_size = 1;
        
        // 只在 Rank 0 或首次调用时打印，避免刷屏 (这里简单处理，打印到 stderr)
        // 实际生产中可以使用 std::call_once
        static bool printed = false;
        if (!printed) {
            std::cerr << "[Config] Loaded: SLICE_SIZE=" << slice_size 
                      << " B, WINDOW=" << window_size 
                      << ", BATCH=" << signal_batch << std::endl;
            printed = true;
        }
    }
};

} // namespace mini_nccl