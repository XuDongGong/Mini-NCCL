cmake_minimum_required(VERSION 3.10)
project(mini-nccl LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "86") # RTX 4060
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 生成动态库必须

# 1. 寻找依赖
find_library(IBVERBS_LIB ibverbs REQUIRED)
find_package(CUDAToolkit REQUIRED) # 引入更完整的 CUDA 支持

# 2. 头文件路径
include_directories(include)
include_directories(src)

# ========================================================
# 3. 构建核心动态库 (libmini_nccl.so)
# ========================================================
set(LIB_SOURCES 
    src/mini_nccl.cu 
    src/api.cpp
)

add_library(mini_nccl SHARED ${LIB_SOURCES})

# 链接依赖
target_link_libraries(mini_nccl PRIVATE ${IBVERBS_LIB} CUDA::cudart)

# ========================================================
# 4. 构建用户测试程序 (User Application)
# ========================================================
# 这个程序模拟 PyTorch，只调用 mini_nccl_api.h，不触碰内部实现
add_executable(app src/main.cpp)

# 链接我们刚刚生成的 mini_nccl 库
target_link_libraries(app mini_nccl CUDA::cudart)