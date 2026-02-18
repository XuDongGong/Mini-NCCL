cmake_minimum_required(VERSION 3.10)
project(mini-nccl LANGUAGES CXX CUDA C)

# ========================================================
# 1. 基础配置
# ========================================================
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 针对 Ubuntu 24.04 (GCC 13) 的兼容性修复
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
include_directories(src/hera) # 确保能找到 Hera 头文件

set(LIB_SOURCES src/mini_nccl.cu src/api.cpp)

add_library(mini_nccl SHARED ${LIB_SOURCES})
target_link_libraries(mini_nccl PRIVATE ${IBVERBS_LIB} CUDA::cudart)

# 设置公共头文件路径
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
# >>> 修复：开启 AVX2 指令集支持 <<<
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(perf_test PRIVATE -mavx2 -mfma)
endif()

# ========================================================
# 5. Hera-Core 相关构建
# ========================================================
# Hera 集成测试
add_executable(hera_test tests/hera_test.cpp)
target_include_directories(hera_test PRIVATE src/hera)

# Hera Master 独立程序
add_executable(hera_master tests/hera_master_main.cpp)
target_include_directories(hera_master PRIVATE src/hera)

# ========================================================
# 6. 安装规则 (Install Rules)
# ========================================================
include(GNUInstallDirs)

install(FILES include/mini_nccl_api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS mini_nccl
    EXPORT MiniNCCLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT MiniNCCLTargets
    FILE MiniNCCLTargets.cmake
    NAMESPACE MiniNCCL::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniNCCL
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/MiniNCCLConfig.cmake 
    "include(\${CMAKE_CURRENT_LIST_DIR}/MiniNCCLTargets.cmake)\n"
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MiniNCCLConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniNCCL
)

message(STATUS "Configured for Install with AVX2 support.")