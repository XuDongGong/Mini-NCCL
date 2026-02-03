#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include "tcp_socket.hpp" // 刚才写的头文件

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

// --- 我们要交换的“名片”结构体 ---
struct RdmaInfo {
    uint32_t qp_num;   // QP 号码
    uint16_t lid;      // Local ID (InfiniBand 地址)
    uint8_t  gid[16];  // Global ID (RoCE IP 地址)
    uint64_t addr;     // 远程内存地址
    uint32_t rkey;     // 远程访问密钥
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./mini_nccl <server|client> [server_ip]" << std::endl;
        return -1;
    }
    bool is_server = (std::string(argv[1]) == "server");
    std::string server_ip = (argc > 2) ? argv[2] : "127.0.0.1";

    std::cout << "=== Mini-NCCL: Dual-Process RDMA ===" << std::endl;

    // 1. 建立 TCP 连接 (用于交换名片)
    TCPSocket tcp;
    if (is_server) {
        tcp.listen_on(8888);
    } else {
        tcp.connect_to(server_ip, 8888);
    }

    // 2. 初始化 RDMA 资源 (和之前一样)
    int num_devices;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    struct ibv_device* rxe_dev = nullptr;
    for (int i=0; i<num_devices; ++i) {
        if (std::string(ibv_get_device_name(dev_list[i])) == "rxe0") {
            rxe_dev = dev_list[i]; break;
        }
    }
    if (!rxe_dev) { std::cerr << "rxe0 not found!" << std::endl; return -1; }

    struct ibv_context* ctx = ibv_open_device(rxe_dev);
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    struct ibv_cq* cq = ibv_create_cq(ctx, 16, nullptr, nullptr, 0);

    // 3. 准备内存
    const size_t size = 1024;
    void* buf = nullptr;
    CUDA_CHECK(cudaHostAlloc(&buf, size, cudaHostAllocDefault));
    
    // 如果我是 Server，我写 "Server says Hi"; 如果我是 Client，我写 "Client says Hello"
    const char* msg = is_server ? "Server says Hi!" : "Client says Hello!";
    strcpy((char*)buf, msg);

    struct ibv_mr* mr = ibv_reg_mr(pd, buf, size, 
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    // 4. 创建 QP
    struct ibv_qp_init_attr qp_attr = {};
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = 10;
    qp_attr.cap.max_recv_wr = 10;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    struct ibv_qp* qp = ibv_create_qp(pd, &qp_attr);

    // 5. 准备我的名片 (Local Info)
    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, 1, &port_attr);
    union ibv_gid my_gid;
    ibv_query_gid(ctx, 1, 1, &my_gid);

    RdmaInfo local_info;
    local_info.qp_num = qp->qp_num;
    local_info.lid = port_attr.lid;
    local_info.addr = (uint64_t)buf;
    local_info.rkey = mr->rkey;
    memcpy(local_info.gid, my_gid.raw, 16);

    // 6. 核心步骤：通过 TCP 交换名片！
    RdmaInfo remote_info;
    // 先发我的，再收对方的
    tcp.send_data(&local_info, sizeof(RdmaInfo));
    tcp.recv_data(&remote_info, sizeof(RdmaInfo));

    std::cout << "[TCP] Exchanged info with peer. Remote QP: " << remote_info.qp_num << std::endl;

    // 7. 使用对方的名片，把 QP 状态机转到底 (INIT -> RTR -> RTS)
    // 注意：这里的 dest_qp_num 现在是 remote_info.qp_num 了！
    
    struct ibv_qp_attr attr = {};
    // -> INIT
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    // -> RTR
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote_info.qp_num; // <--- 填对方的 QP
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.dlid = remote_info.lid;   // <--- 填对方的 LID
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    memcpy(attr.ah_attr.grh.dgid.raw, remote_info.gid, 16); // <--- 填对方的 GID
    attr.ah_attr.grh.sgid_index = 1;
    attr.ah_attr.grh.hop_limit = 1;
    
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);

    // -> RTS
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);

    std::cout << "[RDMA] QP is Ready!" << std::endl;

    // 8. 开始通信：互相 Post Recv，然后 Server 先发，Client 后发
    struct ibv_sge sge;
    sge.addr = (uint64_t)buf;
    sge.length = size;
    sge.lkey = mr->lkey;

    struct ibv_recv_wr recv_wr = {};
    recv_wr.sg_list = &sge;
    recv_wr.num_sge = 1;
    recv_wr.wr_id = 100;
    struct ibv_recv_wr* bad_recv;
    ibv_post_recv(qp, &recv_wr, &bad_recv); // 大家都要先准备好“篮子”接数据

    // 简单做一个同步：等大家都准备好 recv
    char sync = 'G';
    tcp.send_data(&sync, 1);
    tcp.recv_data(&sync, 1);

    // 发送 Send 请求
    struct ibv_send_wr send_wr = {};
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    send_wr.wr_id = 200;
    struct ibv_send_wr* bad_send;
    
    ibv_post_send(qp, &send_wr, &bad_send);

    // 9. 轮询 CQ
    int completions = 0;
    struct ibv_wc wc[2];
    while (completions < 2) {
        int n = ibv_poll_cq(cq, 2, wc);
        for(int i=0; i<n; ++i) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "WC Error: %s\n", ibv_wc_status_str(wc[i].status));
                return -1;
            }
            if (wc[i].opcode == IBV_WC_RECV) {
                std::cout << "[SUCCESS] Received msg from peer: " << (char*)buf << std::endl;
            }
            completions++;
        }
    }

    // 暂停一下让用户看结果
    sleep(1);
    return 0;
}