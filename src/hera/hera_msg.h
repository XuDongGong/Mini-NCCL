#pragma once

#include <cstdint>
#include <cstring>

namespace hera {

// =================================================================
// 1. 基础常量与枚举
// =================================================================

// Magic Number: "HERA" (0x48455241) 
// 防止非法连接干扰 Master (如 SSH 误连)
const uint32_t HERA_MAGIC = 0x48455241;
const uint16_t HERA_VERSION = 1;

// 默认 Master 监听端口
const int HERA_DEFAULT_PORT = 9999;

// 消息类型定义
enum class MessageType : uint8_t {
    REGISTER_REQ   = 0x01, // Worker -> Master: 注册 ("我是 Rank ?")
    REGISTER_RESP  = 0x02, // Master -> Worker: 分配 ("你是 Rank 3")
    
    TOPOLOGY_INIT  = 0x03, // Master -> Worker: 拓扑下发 ("你的右手是 IP:Port")
    TOPOLOGY_ACK   = 0x04, // Worker -> Master: 拓扑就绪 ("我连上右手了")
    
    HEARTBEAT      = 0x05, // Worker -> Master: 心跳 ("我还活着")
    
    GLOBAL_ABORT   = 0xFF  // Master -> Worker: 紧急停车 ("全员停止!")
};

// =================================================================
// 2. 协议头 (固定 12 字节)
// =================================================================
// __attribute__((packed)) 确保没有内存对齐填充，适合网络传输
struct __attribute__((packed)) HeraHeader {
    uint32_t magic;       // 0x48455241
    uint8_t  msg_type;    // MessageType
    uint8_t  version;     // 1
    uint16_t reserved;    // 保留位 (对齐用)
    uint32_t payload_len; // 后续负载的长度
};

// =================================================================
// 3. 具体消息负载 (Payloads)
// =================================================================

// [0x01] REGISTER_REQ: Worker 申请加入
struct __attribute__((packed)) RegisterReq {
    char hostname[64];    // 机器名
    int  pid;             // 进程 ID (用于日志/调试)
};

// [0x02] REGISTER_RESP: Master 分配 Rank
struct __attribute__((packed)) RegisterResp {
    int rank;             // 分配到的 Rank ID
    int world_size;       // 集群总大小
};

// [0x03] TOPOLOGY_INIT: 告诉 Worker 它的 "右手" 是谁
struct __attribute__((packed)) TopologyInfo {
    int prev_rank;
    int next_rank;
    char next_ip[64];     // 下一跳的 IP 地址
    int next_port;        // 下一跳的 RDMA 监听端口
};

// [0x05] HEARTBEAT: 保活
struct __attribute__((packed)) Heartbeat {
    int rank;
    uint32_t state;       // 0: Init, 1: Running, 2: Error
    uint32_t timestamp;   // 简单的逻辑时钟
};

} // namespace hera