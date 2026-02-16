#pragma once
#include "HeraSocket.h"
#include <unistd.h>
#include <iostream>

namespace hera {

class HeraWorker {
    HeraSocket socket_;
    std::string master_ip_;
    int master_port_;
    
    int my_rank_ = -1;
    int world_size_ = -1;
    std::string root_ip_; // 新增: 存储从 Master 获取的 Root IP

public:
    HeraWorker(const std::string& master_ip, int port = HERA_DEFAULT_PORT)
        : master_ip_(master_ip), master_port_(port) {}

    void ConnectAndRegister() {
        try {
            socket_.Connect(master_ip_, master_port_);

            RegisterReq req;
            gethostname(req.hostname, sizeof(req.hostname));
            req.pid = getpid();
            
            socket_.SendMsg(MessageType::REGISTER_REQ, &req, sizeof(req));

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
            root_ip_ = std::string(resp->root_ip); // 解析 Root IP

            std::cout << "[Hera-Worker] Registered! Rank=" << my_rank_ 
                      << "/" << world_size_ << " RootIP=" << root_ip_ << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[Hera-Worker] Init Failed: " << e.what() << std::endl;
            throw; 
        }
    }

    int rank() const { return my_rank_; }
    int size() const { return world_size_; }
    std::string root_ip() const { return root_ip_; } // 提供 Getter
};

} // namespace hera