cmake_minimum_required(VERSION 3.18)
project(cppvllm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# nvcc + libtorch headers are safest at C++17; the kernel doesn't need anything newer
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# GPU architectures for our own .cu files. Override with -DCMAKE_CUDA_ARCHITECTURES=89
# (must be set before the CUDA language is enabled, which find_package(Torch) does)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/W4)
else()
  # CXX only: nvcc does not understand -Wall / -Wextra
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
endif()

# ---- dependencies -------------------------------------------------------------------------
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
find_package(nlohmann_json 3.9 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)

# libtorch from the pip-installed PyTorch:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
# -DCPPVLLM_WITH_TORCH=OFF builds just the parts that do not touch torch.
option(CPPVLLM_WITH_TORCH "Link libtorch" ON)
set(CPPVLLM_HAVE_CUDA_KERNELS OFF)
if(CPPVLLM_WITH_TORCH)
  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")   # carries the matching C++ ABI flag

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER AND TARGET torch_cuda)
    enable_language(CUDA)
    # nvcc's host compile must use the same libstdc++ ABI as libtorch, or std::string symbols fail to link
    string(APPEND CMAKE_CUDA_FLAGS " ${TORCH_CXX_FLAGS}")
    set(CPPVLLM_HAVE_CUDA_KERNELS ON)
  endif()
endif()

# ---- library ------------------------------------------------------------------------------
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS *.cpp engine/*.cpp utils/*.cpp model/*.cpp)
list(FILTER LIB_SOURCES EXCLUDE REGEX "/(example|bench|test)\\.cpp$")

set(KERNEL_SOURCES "")
if(CPPVLLM_HAVE_CUDA_KERNELS)
  file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS kernels/*.cu)
endif()

add_library(cppvllm STATIC ${LIB_SOURCES} ${KERNEL_SOURCES})

target_include_directories(cppvllm PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/engine
  ${CMAKE_SOURCE_DIR}/model
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cppvllm PUBLIC nlohmann_json::nlohmann_json PkgConfig::PCRE2)

if(CPPVLLM_WITH_TORCH)
  target_link_libraries(cppvllm PUBLIC ${TORCH_LIBRARIES})
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_link_libraries(cppvllm PUBLIC CUDA::cudart)   # model_runner.cpp calls cudaMemGetInfo
  endif()
endif()

if(CPPVLLM_HAVE_CUDA_KERNELS)
  # model_architecture.cpp / model_runner.cpp switch to the fused kernel only when this is defined
  target_compile_definitions(cppvllm PUBLIC CPPVLLM_HAS_CUDA_KERNELS=1)
endif()

# ---- programs -----------------------------------------------------------------------------
foreach(app example bench)
  if(EXISTS ${CMAKE_SOURCE_DIR}/${app}.cpp)
    add_executable(${app} ${app}.cpp)
    target_link_libraries(${app} PRIVATE cppvllm)
  endif()
endforeach()

# Throwaway test file. Delete test.cpp when you are done and this block switches itself off.
if(EXISTS ${CMAKE_SOURCE_DIR}/test.cpp)
  add_executable(scratch test.cpp)
  target_compile_options(scratch PRIVATE -UNDEBUG)
  target_link_libraries(scratch PRIVATE cppvllm)
endif()