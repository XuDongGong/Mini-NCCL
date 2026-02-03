#pragma once
#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

// 一个简单的 TCP Socket 包装类，用于交换 RDMA Info
class TCPSocket {
public:
    int sockfd = -1;

    // 1. 作为服务端启动 (Server)
    void listen_on(int port) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("Bind failed"); exit(1);
        }
        listen(sockfd, 1);
        std::cout << "[TCP] Waiting for client on port " << port << "..." << std::endl;

        int new_fd = accept(sockfd, nullptr, nullptr);
        if (new_fd < 0) { perror("Accept failed"); exit(1); }
        
        close(sockfd); // 既然连上了，旧的监听口就不要了
        sockfd = new_fd;
        std::cout << "[TCP] Connection established!" << std::endl;
    }

    // 2. 作为客户端连接 (Client)
    void connect_to(const std::string& ip, int port) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

        // 简单的重试逻辑
        while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cout << "[TCP] Connecting to " << ip << "..." << std::endl;
            sleep(1);
        }
        std::cout << "[TCP] Connected to server!" << std::endl;
    }

    // 3. 发送任意数据
    void send_data(void* data, size_t size) {
        size_t total = 0;
        while (total < size) {
            int n = write(sockfd, (char*)data + total, size - total);
            if (n <= 0) { perror("Write failed"); exit(1); }
            total += n;
        }
    }

    // 4. 接收任意数据
    void recv_data(void* data, size_t size) {
        size_t total = 0;
        while (total < size) {
            int n = read(sockfd, (char*)data + total, size - total);
            if (n <= 0) { perror("Read failed"); exit(1); }
            total += n;
        }
    }

    ~TCPSocket() {
        if (sockfd >= 0) close(sockfd);
    }
};