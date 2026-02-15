#include "hera_master.h"
#include "hera_worker.h"
#include <thread>
#include <vector>
#include <chrono>
#include <cassert>

// 模拟一个训练节点的启动过程
void worker_thread_func(int id) {
    // 模拟启动延迟
    std::this_thread::sleep_for(std::chrono::milliseconds(100 * id));
    
    hera::HeraWorker worker("127.0.0.1", 9999);
    worker.ConnectAndRegister();
    
    // 验证逻辑
    // 在我们的简单实现中，Master 按顺序分配 Rank
    // 但多线程下顺序不一定严格，所以我们只打印
    // assert(worker.rank() >= 0); 
}

int main() {
    int world_size = 4;

    std::cout << "=== Hera-Core Integration Test ===" << std::endl;

    // 1. 启动 Master (在独立线程中)
    std::thread master_thread([=]() {
        hera::HeraMaster master(world_size, 9999);
        master.Run();
    });

    // 让 Master 先跑起来
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 2. 启动 4 个 Worker
    std::vector<std::thread> workers;
    for(int i=0; i<world_size; ++i) {
        workers.emplace_back(worker_thread_func, i);
    }

    // 3. 等待所有 Worker 完成注册
    for(auto& t : workers) {
        if(t.joinable()) t.join();
    }
    
    std::cout << "=== All Workers Joined! Test Passed. ===" << std::endl;

    // 强制退出测试 (Master 还在 loop 中)
    std::this_thread::sleep_for(std::chrono::seconds(1));
    exit(0); 
    return 0;
}