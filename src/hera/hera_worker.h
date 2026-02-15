#pragma once
#include "HeraSocket.h"
#include <unistd.h>
#include <iostream>

namespace hera {

class HeraWorker {
    HeraSocket socket_;
    std::string master_ip_;
    int master_port_;
    
    // 我的身份信息
    int my_rank_ = -1;
    int world_size_ = -1;

public:
    HeraWorker(const std::string& master_ip, int port = HERA_DEFAULT_PORT)
        : master_ip_(master_ip), master_port_(port) {}

    // 连接并注册
    void ConnectAndRegister() {
        try {
            // 1. 连接 Master
            socket_.Connect(master_ip_, master_port_);

            // 2. 构造注册请求
            RegisterReq req;
            gethostname(req.hostname, sizeof(req.hostname));
            req.pid = getpid();
            
            // 3. 发送
            socket_.SendMsg(MessageType::REGISTER_REQ, &req, sizeof(req));

            // 4. 等待回复 (阻塞)
            MessageType type;
            std::vector<uint8_t> buffer;
            if (!socket_.RecvMsg(type, buffer)) {
                throw std::runtime_error("Master closed connection");
            }

            if (type != MessageType::REGISTER_RESP) {
                throw std::runtime_error("Unexpected response from Master");
            }

            auto* resp = (RegisterResp*)buffer.data();
            my_rank_ = resp->rank;
            world_size_ = resp->world_size;

            std::cout << "[Hera-Worker] Registered successfully! Rank=" << my_rank_ 
                      << "/" << world_size_ << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[Hera-Worker] Init Failed: " << e.what() << std::endl;
            throw; // 继续抛出，让 Mini-NCCL 知道初始化失败
        }
    }

    int rank() const { return my_rank_; }
    int size() const { return world_size_; }
};

} // namespace hera