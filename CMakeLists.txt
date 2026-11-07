cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "CUDA architectures")   # must be set before enabling CUDA
project(cppvllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "cppvllm build type: ${CMAKE_BUILD_TYPE}")

# ---- dependencies -------------------------------------------------------------------------
find_package(nlohmann_json 3.9 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)


set(_NCCL_PIP_DIR "${Torch_DIR}/../../../../nvidia/nccl")
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS $ENV{NCCL_HOME} ${_NCCL_PIP_DIR} $ENV{CUDA_HOME} ${CUDAToolkit_ROOT}
  PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2
  HINTS $ENV{NCCL_HOME} ${_NCCL_PIP_DIR}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL headers/library not found. Install the NVIDIA NCCL development package or set NCCL_HOME.")
endif()
message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
message(STATUS "NCCL library: ${NCCL_LIBRARY}")

string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
string(APPEND CMAKE_CUDA_FLAGS " ${TORCH_CXX_FLAGS}")

include(FetchContent)
FetchContent_Declare(flash_attn
  GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
  GIT_TAG        main-0.8.2.post1
  GIT_SHALLOW    TRUE
  GIT_SUBMODULES csrc/cutlass)
FetchContent_MakeAvailable(flash_attn)

set(FA_DIR "${flash_attn_SOURCE_DIR}/csrc/flash_attn")

set(FA_KERNELS
  ${FA_DIR}/src/flash_fwd_hdim128_bf16_sm80.cu
  ${FA_DIR}/src/flash_fwd_hdim128_bf16_causal_sm80.cu
  ${FA_DIR}/src/flash_fwd_split_hdim128_bf16_sm80.cu
  ${FA_DIR}/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu)

# ---- library ------------------------------------------------------------------------------
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS *.cpp engine/*.cpp utils/*.cpp kernels/*.cpp model/*.cpp)
list(FILTER LIB_SOURCES EXCLUDE REGEX "/(example|bench|test[a-z_]*)\\.cpp$")

add_library(cppvllm STATIC
  ${LIB_SOURCES}
  kernels/fused_ops.cu
  kernels/paged_attention.cu
  ${FA_DIR}/flash_api.cpp
  ${FA_KERNELS})

set_source_files_properties(${FA_DIR}/flash_api.cpp PROPERTIES COMPILE_OPTIONS "-w")   

target_include_directories(cppvllm
  PUBLIC  ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/engine ${CMAKE_SOURCE_DIR}/utils ${CMAKE_SOURCE_DIR}/model
  PUBLIC  ${NCCL_INCLUDE_DIR}
  PRIVATE ${FA_DIR} ${FA_DIR}/src ${flash_attn_SOURCE_DIR}/csrc/cutlass/include)

target_link_libraries(cppvllm PUBLIC
  nlohmann_json::nlohmann_json
  PkgConfig::PCRE2
  ${TORCH_LIBRARIES}
  CUDA::cudart
  ${NCCL_LIBRARY}
  Threads::Threads)

target_compile_definitions(cppvllm PRIVATE
  FLASHATTENTION_DISABLE_PYBIND
  FLASHATTENTION_DISABLE_BACKWARD
  FLASHATTENTION_DISABLE_DROPOUT
  FLASHATTENTION_DISABLE_ALIBI
  FLASHATTENTION_DISABLE_SOFTCAP
  FLASHATTENTION_DISABLE_LOCAL)

target_compile_options(cppvllm PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__>")

# ---- programs -----------------------------------------------------------------------------
foreach(app example bench)
  add_executable(${app} ${app}.cpp)
  target_link_libraries(${app} PRIVATE cppvllm)
endforeach()