# Mini-NCCL

Mini-NCCL 是一个用于面试/学习的 **NCCL 风格集合通信原型**：在多进程场景下实现 **Ring AllReduce（scatter-reduce + all-gather）**，数据面采用 **RDMA Write 单边语义 + flag 协议**，同机可自动尝试 **CUDA IPC 快路**；同时包含一套轻量控制面 **Hera-Core**（自动 rank 分配与引导组网）。

本仓库的核心价值是：把通信库最关键的工程要点（RDMA verbs、MR cache、slice pipeline、NUMA affinity、GPU-side wait + watchdog、可观测性 NVTX、Hera 控制面协议等）用较小代码量跑通，并配套大量“面试可复述”的学习文档（见下方文档索引）。

---

## 目录结构

```text
include/                 # 对外 C API + 内部核心抽象
src/
  api.cpp                # C ABI 桥接层（异常防火墙）+ NVTX + CUDA Graph capture 感知 + Hera 集成
  mini_nccl.cu           # Ring AllReduce + slice pipeline + GPU wait kernel + watchdog/abort
  transport/
    RDMATransport.h      # verbs RDMA write + MR cache + request pool + 智能网卡选择 + NUMA 绑定 + IPC 尝试
    LockFreeQueue.h      # SPSC 无锁队列（用于 request pool 索引管理）
    Socket.h             # RDMA bootstrap 用的 TCP socket（8888）
  hera/
    hera_master.h        # Hera Master（控制面）
    hera_worker.h        # Hera Worker（嵌入式 agent）
    hera_msg.h           # Hera 协议头（magic/version/type/len）与消息结构
    HeraSocket.h         # Hera 的 TLV framing + magic/version 校验
tests/
  perf_test.cpp          # 性能基准（含 AVX2 SIMD 校验）+ 支持 rank=-1 Hera 自动组网
  hera_test.cpp          # Hera master/worker 集成测试
  hera_master_main.cpp   # 独立运行 Hera master 的入口
demos/01_dual_process_rdma/
  README.md              # Soft-RoCE 双进程 RDMA demo
docs/                    # 面试/学习文档（大量 P8/P9 深挖点）
```

---

## 核心特性（按“面试可讲”组织）

- **算法**：Ring AllReduce 两阶段（scatter-reduce + all-gather），按 slice 流水推进
- **数据面传输**：
  - 跨机：verbs `IBV_WR_RDMA_WRITE`（数据）+ RDMA write inline（flag）
  - 同机：CUDA IPC 映射后 `cudaMemcpyDeviceToDevice` 快路 + 直接写对端 flag
- **进度/同步**：GPU-side `wait_kernel` 轮询 flag，host watchdog 超时注入 `abort_flag` 做 bounded-fail
- **性能工程**：
  - **Zero-Allocation**：`Context` 冷启动预分配双缓冲 `cudaHostAllocMapped`，热路径 \(O(1)\) 指针偏移取 buffer
  - **MR Cache**：`unordered_map<void*, MR>` 缓存 `ibv_reg_mr`，利用地址复用消除毫秒级注册开销
  - **Request Pool**：预分配 `RDMARequest` 池 + 无锁队列管理索引，避免 `new/delete`
  - **Selective signaling + window**：降低 CQ 压力
  - **智能网卡选择 + NUMA 绑定**：环境变量 override + 启发式选择 + 自动绑核
- **可观测性**：NVTX range（配合 Nsight Systems 对齐 CPU/GPU 时间线）
- **CUDA Graph 感知**：`cudaStreamIsCapturing` 检测 capture 并告警
- **控制面（Hera-Core）**：自定义二进制协议（magic/version/type/len），rank=-1 自动注册分配 rank/size/root_ip

---

## 构建

依赖：

- CUDA Toolkit
- RDMA verbs（`libibverbs`）
- NVTX（`nvToolsExt`）

编译：

```bash
mkdir -p build
cd build
cmake ..
make -j
```

产物：

- `libmini_nccl.so`：核心库
- `app`：最小示例
- `perf_test`：性能基准（含 AVX2）
- `hera_master` / `hera_test`：Hera 控制面相关

---

## 运行示例

### 1) 最小示例（2 rank）

开两个终端：

```bash
./build/app 0 127.0.0.1
./build/app 1 127.0.0.1
```

### 2) 性能基准（推荐）

开两个终端（手动 rank）：

```bash
./build/perf_test 0 2 127.0.0.1
./build/perf_test 1 2 127.0.0.1
```

启用 Hera 自动组网（rank=-1）：

```bash
./build/hera_master 2

./build/perf_test -1 2 127.0.0.1
./build/perf_test -1 2 127.0.0.1
```

> 说明：Hera 模式下，`perf_test` 的第三个参数会被当作 Hera Master 的 IP。

---

## 配置（环境变量）

- **`MINI_NCCL_NET_DEVICE`**：指定 RDMA 设备名（例如 `mlx5_0`、`rxe0`）
- **`MINI_NCCL_SLICE_SIZE`**：切片大小（字节），默认 `128*1024`
- **`MINI_NCCL_WINDOW_SIZE`**：窗口大小（默认 64）
- **`MINI_NCCL_SIGNAL_BATCH`**：信号批次（默认 16）

示例：

```bash
export MINI_NCCL_NET_DEVICE=mlx5_0
export MINI_NCCL_SLICE_SIZE=$((256*1024))
export MINI_NCCL_WINDOW_SIZE=32
export MINI_NCCL_SIGNAL_BATCH=8
```

---

## 文档索引（强烈建议从这里读）

- **总讲解稿（五层讲透）**：`docs/P8P9_面试讲解稿_五层讲透MiniNCCL.md`
- **Memory Consistency（外设写入可见性）**：`docs/P8P9_面试挖掘点_03_Memory_Consistency_内存一致性.md`
- **GPUDirect RDMA 故事线**：`docs/P8P9_面试故事线_GPUDirect_RDMA_从WSL2到H800的零拷贝路径.md`
- **GPUDirect 架构支持（概念/源码/收益/面试）**：`docs/P8P9_GPUDirect_架构支持_概念-源码-收益-面试实战.md`
- **NUMA 绑定**：`docs/P8P9_面试挖掘点_04_NUMA_架构感知与CPU亲和性绑定.md`
- **SIMD 校验（AVX2）**：`docs/P8P9_面试挖掘点_05_SIMD_AVX2_极速数据校验.md`
- **SIMD（AVX2）指令级拆解 + 面试问答**：`docs/P8P9_面试挖掘点_05b_SIMD_AVX2_从Intrinsics到指令_本项目应用与面试问答.md`
- **无锁编程（SPSC 队列）**：`docs/P8P9_面试挖掘点_06_LockFree_无锁队列与并发内存模型.md`
- **分支预测/I-Cache**：`docs/P8P9_面试挖掘点_07_Branch_Prediction_分支预测与I-Cache优化.md`
- **可观测性/生产适配**：`docs/P8P9_面试挖掘点_08_Observability_Production_Readiness_可观测性与生产适配.md`
- **CUDA Graph capture 感知**：`docs/P8P9_面试挖掘点_09_CUDA_Graph_Stream_Capture_感知与防御.md`
- **MR Cache**：`docs/P8P9_面试挖掘点_10_MR_Cache_显存注册缓存与零开销寻址.md`
- **魔术数/TLV 协议头（Hera 协议基础）**：`docs/P8P9_面试挖掘点_11_Magic_Number_TCP协议头与控制面鲁棒性.md`
- **智能网卡选择**：`docs/P8P9_面试挖掘点_12_Smart_NIC_Selection_智能网卡选择与环境自适应.md`
- **Zero-Allocation（内存复用）**：`docs/P8P9_面试挖掘点_13_Zero_Allocation_内存池化与内存复用机制.md`
- **Hera-Core 超详细学习文档**：`docs/P8P9_面试挖掘点_14_Hera_Core_控制平面学习文档_从0到面试讲清楚.md`
- **C API 桥接：non-owning `shared_ptr` 与生命周期治理**：`docs/P8P9_面试挖掘点_15_C_API_Bridge_non_owning_shared_ptr_裸指针桥接与生命周期治理.md`

---

## 已知限制（当前实现边界）

- **Hera-Core**：当前版本主要实现了 REGISTER（自动 rank/size/root_ip 注入）；`HEARTBEAT/GLOBAL_ABORT/TOPOLOGY_*` 消息类型已预留但未完整落地为运行时 fail-fast。
- **MR Cache**：当前是 map cache（无 LRU/水位线），更偏原型；生产级需要失效策略与 pinned 资源上限。
- **CUDA Graph**：当前仅做 capture 感知与告警，整体 transport 仍依赖 host 控制面，不保证 graph-safe。

---

## License

未在仓库中声明许可证（如需开源发布，建议补充 `LICENSE`）。

