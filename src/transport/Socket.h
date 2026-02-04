#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

namespace mini_nccl {

// 代表一个已建立的 TCP 连接
class Socket {
public:
    int fd_ = -1;

    Socket(int fd) : fd_(fd) {
        // 禁用 Nagle 算法，降低 TCP 延迟
        int flag = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int));
    }

    ~Socket() {
        if (fd_ != -1) close(fd_);
    }

    // 发送固定长度数据 (阻塞直到发完)
    void send(const void* data, size_t size) {
        size_t sent = 0;
        const char* ptr = static_cast<const char*>(data);
        while (sent < size) {
            ssize_t ret = ::write(fd_, ptr + sent, size - sent);
            if (ret <= 0) throw std::runtime_error("Socket send failed");
            sent += ret;
        }
    }

    // 接收固定长度数据 (阻塞直到收满)
    void recv(void* data, size_t size) {
        size_t received = 0;
        char* ptr = static_cast<char*>(data);
        while (received < size) {
            ssize_t ret = ::read(fd_, ptr + received, size - received);
            if (ret <= 0) throw std::runtime_error("Socket recv failed (Connection closed?)");
            received += ret;
        }
    }
};

// 代表服务端监听 Socket
class ServerSocket {
public:
    int fd_ = -1;

    ServerSocket(int port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) throw std::runtime_error("Failed to create socket");

        int opt = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (::bind(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("Bind failed. Port already in use?");
        }
        if (::listen(fd_, 10) < 0) {
            throw std::runtime_error("Listen failed");
        }
    }

    ~ServerSocket() {
        if (fd_ != -1) close(fd_);
    }

    // 接受一个新连接，返回 Socket 对象
    std::shared_ptr<Socket> accept() {
        int conn_fd = ::accept(fd_, nullptr, nullptr);
        if (conn_fd < 0) throw std::runtime_error("Accept failed");
        return std::make_shared<Socket>(conn_fd);
    }
};

// 客户端连接辅助函数
inline std::shared_ptr<Socket> connect_to(std::string ip, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("Failed to create socket");

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

    // 简单的重试逻辑
    int retries = 0;
    while (::connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        if (++retries > 20) throw std::runtime_error("Failed to connect to " + ip);
        sleep(1); // 等待 Server 启动
    }
    return std::make_shared<Socket>(fd);
}

} // namespace mini_nccl