#pragma once
#include "hera_msg.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

namespace hera {

class HeraSocket {
    int fd_ = -1;

public:
    HeraSocket(int fd = -1) : fd_(fd) {}
    ~HeraSocket() { if (fd_ > 0) close(fd_); }

    HeraSocket(const HeraSocket&) = delete;
    HeraSocket& operator=(const HeraSocket&) = delete;
    HeraSocket(HeraSocket&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    HeraSocket& operator=(HeraSocket&& other) noexcept {
        if (this != &other) {
            if (fd_ > 0) close(fd_);
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    int fd() const { return fd_; }
    bool is_valid() const { return fd_ > 0; }

    // 获取对端 IP 地址 (新增!)
    std::string GetPeerIP() const {
        if (fd_ < 0) return "";
        struct sockaddr_in addr;
        socklen_t len = sizeof(addr);
        if (getpeername(fd_, (struct sockaddr*)&addr, &len) == 0) {
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
            return std::string(ip);
        }
        return "";
    }

    void Connect(const std::string& ip, int port) {
        if (fd_ > 0) close(fd_);
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) throw std::runtime_error("Socket creation failed");

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

        if (connect(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("Hera Connect failed to " + ip);
        }
    }

    void Listen(int port) {
        if (fd_ > 0) close(fd_);
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) throw std::runtime_error("Hera Bind failed");
        if (listen(fd_, 10) < 0) throw std::runtime_error("Hera Listen failed");
    }

    std::unique_ptr<HeraSocket> Accept() {
        int conn_fd = accept(fd_, nullptr, nullptr);
        if (conn_fd < 0) throw std::runtime_error("Hera Accept failed");
        return std::make_unique<HeraSocket>(conn_fd);
    }

    void SendMsg(MessageType type, const void* payload, uint32_t len) {
        HeraHeader header;
        header.magic = HERA_MAGIC;
        header.msg_type = (uint8_t)type;
        header.version = HERA_VERSION;
        header.reserved = 0;
        header.payload_len = len;
        SendRaw(&header, sizeof(header));
        if (len > 0) SendRaw(payload, len);
    }

    bool RecvMsg(MessageType& type, std::vector<uint8_t>& payload) {
        HeraHeader header;
        if (!RecvRaw(&header, sizeof(header))) return false;
        if (header.magic != HERA_MAGIC) throw std::runtime_error("Invalid Magic Number");
        if (header.version != HERA_VERSION) throw std::runtime_error("Version Mismatch");
        type = (MessageType)header.msg_type;
        payload.resize(header.payload_len);
        if (header.payload_len > 0) {
            if (!RecvRaw(payload.data(), header.payload_len)) return false;
        }
        return true;
    }

private:
    void SendRaw(const void* data, size_t len) {
        const uint8_t* ptr = (const uint8_t*)data;
        size_t sent = 0;
        while (sent < len) {
            ssize_t ret = write(fd_, ptr + sent, len - sent);
            if (ret < 0) throw std::runtime_error("Socket Write Error");
            sent += ret;
        }
    }

    bool RecvRaw(void* data, size_t len) {
        uint8_t* ptr = (uint8_t*)data;
        size_t received = 0;
        while (received < len) {
            ssize_t ret = read(fd_, ptr + received, len - received);
            if (ret < 0) throw std::runtime_error("Socket Read Error");
            if (ret == 0) return false; 
            received += ret;
        }
        return true;
    }
};

} // namespace hera