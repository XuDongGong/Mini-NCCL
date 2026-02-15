cmake_minimum_required(VERSION 3.10)
project(mini-nccl LANGUAGES CXX CUDA C)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 2. 针对 Ubuntu 24.04 (GCC 13) 的唯一必要 Hack
# 只要不开启 C++17，通常不需要手动定义 _Float32
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -O3")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 3. 寻找依赖
find_library(IBVERBS_LIB ibverbs REQUIRED)
find_package(CUDAToolkit REQUIRED) # 保持引入完整支持

# 4. 头文件路径
include_directories(include)
include_directories(src)

# ========================================================
# 5. 构建核心动态库 (libmini_nccl.so)
# ========================================================
set(LIB_SOURCES 
    src/mini_nccl.cu 
    src/api.cpp
)

add_library(mini_nccl SHARED ${LIB_SOURCES})
# 现代写法：链接 CUDA::cudart 而不是手动 link
target_link_libraries(mini_nccl PRIVATE ${IBVERBS_LIB} CUDA::cudart)

# ========================================================
# 6. 构建用户测试程序 (App)
# ========================================================
add_executable(app src/main.cpp)
target_link_libraries(app mini_nccl CUDA::cudart)

# ========================================================
# 7. 构建性能基准测试 (Perf Test) - NEW!
# ========================================================
# 注意：perf_test.cpp 里面有 CUDA 代码吗？
# 如果 tests/perf_test.cpp 只是调用 API，用 add_executable 即可
# 但为了安全链接，我们显式链接 CUDA
add_executable(perf_test tests/perf_test.cpp)
target_link_libraries(perf_test mini_nccl CUDA::cudart)