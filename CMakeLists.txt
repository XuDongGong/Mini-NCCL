cmake_minimum_required(VERSION 3.10)
project(mini-nccl LANGUAGES CXX CUDA C)

# ========================================================
# 1. 基础配置 (回归最稳的 C++14)
# ========================================================
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 针对 Ubuntu 24.04 (GCC 13) 的唯一必要 Hack
# 我们删除了导致冲突的 _Float 定义，只保留允许新编译器
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -O3")
endif()

# ========================================================
# 2. 依赖查找
# ========================================================
find_package(CUDAToolkit REQUIRED)
find_library(IBVERBS_LIB ibverbs REQUIRED)

# ========================================================
# 3. 构建核心库 (libmini_nccl.so)
# ========================================================
include_directories(include)
include_directories(src)

set(LIB_SOURCES src/mini_nccl.cu src/api.cpp)

add_library(mini_nccl SHARED ${LIB_SOURCES})
target_link_libraries(mini_nccl PRIVATE ${IBVERBS_LIB} CUDA::cudart)

# 设置公共头文件路径 (供安装使用)
target_include_directories(mini_nccl PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ========================================================
# 4. 构建工具 (App & Benchmark)
# ========================================================
add_executable(app src/main.cpp)
target_link_libraries(app mini_nccl CUDA::cudart)

add_executable(perf_test tests/perf_test.cpp)
target_link_libraries(perf_test mini_nccl CUDA::cudart)

# ========================================================
# 5. 安装规则 (Install Rules)
# ========================================================
include(GNUInstallDirs)

# 安装头文件
install(FILES include/mini_nccl_api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# 安装动态库
install(TARGETS mini_nccl
    EXPORT MiniNCCLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 生成并安装 CMake 配置文件 (供 find_package 使用)
install(EXPORT MiniNCCLTargets
    FILE MiniNCCLTargets.cmake
    NAMESPACE MiniNCCL::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniNCCL
)

# 生成版本文件等 (简化版)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/MiniNCCLConfig.cmake 
    "include(\${CMAKE_CURRENT_LIST_DIR}/MiniNCCLTargets.cmake)\n"
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MiniNCCLConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniNCCL
)

message(STATUS "Configured for Install. Usage: sudo make install")