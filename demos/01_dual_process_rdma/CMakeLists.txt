cmake_minimum_required(VERSION 3.10)
project(mini-nccl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 自动寻找 RDMA 库
find_library(IBVERBS_LIB ibverbs REQUIRED)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 编译主程序
add_executable(mini_nccl src/main.cu)

# 链接 RDMA 库
target_link_libraries(mini_nccl PRIVATE ${IBVERBS_LIB})

# 设置显卡架构
set_target_properties(mini_nccl PROPERTIES CUDA_ARCHITECTURES "86")