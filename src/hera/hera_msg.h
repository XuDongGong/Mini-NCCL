#pragma once
#include <cstdint>
#include <cstring>

namespace hera {

const uint32_t HERA_MAGIC = 0x48455241;
const uint16_t HERA_VERSION = 1;
const int HERA_DEFAULT_PORT = 9999;

enum class MessageType : uint8_t {
    REGISTER_REQ   = 0x01, 
    REGISTER_RESP  = 0x02, 
    TOPOLOGY_INIT  = 0x03, 
    TOPOLOGY_ACK   = 0x04, 
    HEARTBEAT      = 0x05, 
    GLOBAL_ABORT   = 0xFF  
};

struct __attribute__((packed)) HeraHeader {
    uint32_t magic;
    uint8_t  msg_type;
    uint8_t  version;
    uint16_t reserved;
    uint32_t payload_len;
};

struct __attribute__((packed)) RegisterReq {
    char hostname[64];    
    int  pid;             
};

struct __attribute__((packed)) RegisterResp {
    int rank;             
    int world_size;       
    char root_ip[64];     // 新增: Rank 0 的 IP 地址，用于引导 RDMA 组网
};

struct __attribute__((packed)) TopologyInfo {
    int prev_rank;
    int next_rank;
    char next_ip[64];     
    int next_port;        
};

struct __attribute__((packed)) Heartbeat {
    int rank;
    uint32_t state;       
    uint32_t timestamp;   
};

} // namespace hera