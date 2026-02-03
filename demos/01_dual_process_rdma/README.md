# Demo 01: Dual-Process RDMA All-Reduce (Soft-RoCE)

这是一个基于 Soft-RoCE (RXE) 的 RDMA 通信原型验证项目。
它演示了如何在无物理 RDMA 网卡的环境下，利用 WSL2 定制内核实现 GPU (CUDA) 配合 RDMA 进行双进程集合通信 (All-Reduce Sum)。

## 🎯 功能特性
- **TCP Bootstrap**: 使用 TCP Socket 交换 RDMA QP 信息。
- **Pinned Memory**: 使用 `cudaHostAlloc` 分配锁页内存。
- **QP State Machine**: 完整演示 QP 从 RESET -> INIT -> RTR -> RTS 的状态流转。
- **RDMA Write**: 使用单边写入 (One-sided Write) 传输数据。
- **CUDA Compute**: 使用 GPU 核函数进行向量加法。

## 🚀 编译与运行
```bash
mkdir build && cd build
cmake ..
make -j8
./mini_nccl 0  # Terminal 1 (Server)
./mini_nccl 1  # Terminal 2 (Client)
```
