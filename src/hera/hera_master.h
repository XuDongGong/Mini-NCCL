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
    std::string ip; // 新增: 记录 IP
    int pid;
    int rank;
};

class HeraMaster {
    HeraSocket server_socket_;
    int world_size_;
    int port_;
    std::atomic<bool> running_{true};
    
    std::vector<std::unique_ptr<HeraSocket>> worker_conns_;
    std::vector<WorkerContext> worker_infos_;

public:
    HeraMaster(int world_size, int port = HERA_DEFAULT_PORT) 
        : world_size_(world_size), port_(port) {}

    void Run() {
        try {
            server_socket_.Listen(port_);
            std::cout << "[Hera-Master] Listening on port " << port_ 
                      << ", waiting for " << world_size_ << " workers..." << std::endl;

            while (worker_conns_.size() < (size_t)world_size_) {
                auto client = server_socket_.Accept();
                HandleRegister(std::move(client));
            }

            std::cout << "[Hera-Master] All " << world_size_ << " workers registered! Cluster is Ready." << std::endl;
            std::cout << "[Hera-Master] Root IP (Rank 0) is: " << worker_infos_[0].ip << std::endl;

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
        std::string peer_ip = client->GetPeerIP();

        if (!client->RecvMsg(type, buffer)) {
            std::cerr << "[Hera-Master] Client disconnected during registration" << std::endl;
            return;
        }

        if (type != MessageType::REGISTER_REQ) {
            std::cerr << "[Hera-Master] Unexpected msg type" << std::endl;
            return;
        }

        auto* req = (RegisterReq*)buffer.data();
        
        int rank = worker_conns_.size();
        
        WorkerContext ctx;
        ctx.fd = client->fd();
        ctx.rank = rank;
        ctx.pid = req->pid;
        ctx.hostname = req->hostname;
        ctx.ip = peer_ip; // 记录 IP
        worker_infos_.push_back(ctx);

        std::cout << "[Hera-Master] New Worker: Rank=" << rank 
                  << " Host=" << ctx.hostname << " IP=" << ctx.ip << " PID=" << ctx.pid << std::endl;

        // 构建回复
        RegisterResp resp;
        resp.rank = rank;
        resp.world_size = world_size_;
        
        // 核心逻辑: 填充 Rank 0 的 IP
        // 如果我是 Rank 0，那 Root IP 就是我自己
        // 如果我是 Rank N，那 Root IP 是列表里第一个人的 IP
        std::string root_ip = worker_infos_[0].ip;
        strncpy(resp.root_ip, root_ip.c_str(), 63);

        client->SendMsg(MessageType::REGISTER_RESP, &resp, sizeof(resp));
        worker_conns_.push_back(std::move(client));
    }
};

} // namespace hera