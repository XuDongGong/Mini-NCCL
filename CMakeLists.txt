cmake_minimum_required(VERSION 3.10)
project(mini-nccl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "86") # RTX 4060

# 1. 寻找依赖
find_library(IBVERBS_LIB ibverbs REQUIRED)

# 2. 包含头文件
include_directories(include)
include_directories(src)

# 3. 编译主程序
# 注意：这里我们同时编译库实现(.cu)和测试程序(.cpp)
add_executable(mini_nccl 
    src/main.cpp 
    src/mini_nccl.cu
)

# 4. 链接依赖
# cudart 是 CUDA Runtime，ibverbs 是 RDMA
target_link_libraries(mini_nccl PRIVATE ${IBVERBS_LIB} cudart)