cmake_minimum_required(VERSION 3.24)

project(nvidia_of_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

if(NOT DEFINED NVOF_SDK_ROOT)
    file(GLOB SDK_ROOT_CANDIDATES LIST_DIRECTORIES true "${REPO_ROOT}/packages/Optical_Flow_SDK*")
    list(SORT SDK_ROOT_CANDIDATES ORDER DESCENDING)
    list(LENGTH SDK_ROOT_CANDIDATES SDK_ROOT_CANDIDATE_COUNT)
    if(SDK_ROOT_CANDIDATE_COUNT GREATER 0)
        list(GET SDK_ROOT_CANDIDATES 0 NVOF_SDK_ROOT)
    else()
        set(NVOF_SDK_ROOT "${REPO_ROOT}/packages/Optical_Flow_SDK_5.0.7")
    endif()
endif()

if(NOT EXISTS "${NVOF_SDK_ROOT}/NvOFInterface/nvOpticalFlowCommon.h")
    message(FATAL_ERROR "NVOF_SDK_ROOT does not point to a valid Optical Flow SDK root: ${NVOF_SDK_ROOT}")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(nvidia_of_bridge SHARED
    nvof_bridge.cpp
    nvof_utils_stub.cpp
    nvof_bridge_kernels.cu
    ${NVOF_SDK_ROOT}/Common/NvOFBase/NvOF.cpp
    ${NVOF_SDK_ROOT}/Common/NvOFBase/NvOFCuda.cpp
    ${NVOF_SDK_ROOT}/Common/Utils/NvOFUtilsCuda.cpp
    ${NVOF_SDK_ROOT}/Common/Utils/kernel.cu
)

target_include_directories(nvidia_of_bridge PRIVATE
    ${NVOF_SDK_ROOT}/Common/NvOFBase
    ${NVOF_SDK_ROOT}/Common/Utils
    ${NVOF_SDK_ROOT}/NvOFInterface
)

target_link_libraries(nvidia_of_bridge PRIVATE CUDA::cuda_driver CUDA::cudart)

set_target_properties(nvidia_of_bridge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "nvidia_of_bridge"
)

if(MSVC)
    target_compile_definitions(nvidia_of_bridge PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
