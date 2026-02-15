#pragma once
#include "HeraSocket.h"
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <thread>
#include <atomic>

namespace hera {

struct WorkerContext {
    int fd;
    std::string hostname;
    int pid;
    int rank;
    // 可以在这里扩展: last_heartbeat_time
};

class HeraMaster {
    HeraSocket server_socket_;
    int world_size_;
    int port_;
    std::atomic<bool> running_{true};
    
    // 存储所有连接的 Worker
    std::vector<std::unique_ptr<HeraSocket>> worker_conns_;
    std::vector<WorkerContext> worker_infos_;

public:
    HeraMaster(int world_size, int port = HERA_DEFAULT_PORT) 
        : world_size_(world_size), port_(port) {}

    void Run() {
        try {
            // 1. 启动监听
            server_socket_.Listen(port_);
            std::cout << "[Hera-Master] Listening on port " << port_ 
                      << ", waiting for " << world_size_ << " workers..." << std::endl;

            // 2. 阻塞等待所有 Worker 注册 (Bootstrap Phase)
            // 在这一阶段，我们不需要复杂的 Epoll，只需要一个一个接客
            while (worker_conns_.size() < (size_t)world_size_) {
                auto client = server_socket_.Accept();
                HandleRegister(std::move(client));
            }

            std::cout << "[Hera-Master] All " << world_size_ << " workers registered! Cluster is Ready." << std::endl;

            // 3. (Todo in Sprint 3) 进入 Runtime 循环，处理心跳
            // 现在先简单 sleep 模拟守护
            while(running_) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

        } catch (const std::exception& e) {
            std::cerr << "[Hera-Master] Fatal Error: " << e.what() << std::endl;
        }
    }

    void Stop() { running_ = false; }

private:
    void HandleRegister(std::unique_ptr<HeraSocket> client) {
        MessageType type;
        std::vector<uint8_t> buffer;

        // 1. 读取注册包
        if (!client->RecvMsg(type, buffer)) {
            std::cerr << "[Hera-Master] Client disconnected during registration" << std::endl;
            return;
        }

        if (type != MessageType::REGISTER_REQ) {
            std::cerr << "[Hera-Master] Unexpected msg type during bootstrap" << std::endl;
            return;
        }

        auto* req = (RegisterReq*)buffer.data();
        
        // 2. 分配 Rank (简单策略：按到达顺序)
        int rank = worker_conns_.size();
        
        WorkerContext ctx;
        ctx.fd = client->fd();
        ctx.rank = rank;
        ctx.pid = req->pid;
        ctx.hostname = req->hostname;
        worker_infos_.push_back(ctx);

        std::cout << "[Hera-Master] New Worker: Rank=" << rank 
                  << " Host=" << ctx.hostname << " PID=" << ctx.pid << std::endl;

        // 3. 回复 Rank 信息
        RegisterResp resp;
        resp.rank = rank;
        resp.world_size = world_size_;
        client->SendMsg(MessageType::REGISTER_RESP, &resp, sizeof(resp));

        // 4. 保存连接 (Move ownership)
        worker_conns_.push_back(std::move(client));
    }
};

} // namespace hera