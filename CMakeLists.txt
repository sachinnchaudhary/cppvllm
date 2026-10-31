cmake_minimum_required(VERSION 3.18)
project(cppvllm LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# ---- dependencies -------------------------------------------------------------------------
# nlohmann/json reads config.json and the safetensors header. Use the system copy if there is
# one (sudo apt install nlohmann-json3-dev), otherwise download it at configure time.
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

# libtorch from the pip-installed PyTorch:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
# (A CUDA build of torch needs the matching CUDA toolkit installed so find_package can work.)
# -DCPPVLLM_WITH_TORCH=OFF builds just the parts that do not touch torch.
option(CPPVLLM_WITH_TORCH "Link libtorch" ON)
if(CPPVLLM_WITH_TORCH)
  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")   # carries the matching C++ ABI flag
endif()

# ---- library ------------------------------------------------------------------------------
file(GLOB LIB_SOURCES CONFIGURE_DEPENDS *.cpp engine/*.cpp utils/*.cpp)
list(FILTER LIB_SOURCES EXCLUDE REGEX "/(example|bench|test)\\.cpp$")   # programs, not library code

add_library(cppvllm STATIC ${LIB_SOURCES})

# FIX: Added ${TORCH_INCLUDE_DIRS} when building with Torch
if(CPPVLLM_WITH_TORCH)
  target_include_directories(cppvllm PUBLIC 
    ${CMAKE_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/engine
    ${TORCH_INCLUDE_DIRS}
  )
else()
  target_include_directories(cppvllm PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/engine)
endif()

target_link_libraries(cppvllm PUBLIC nlohmann_json::nlohmann_json)
if(CPPVLLM_WITH_TORCH)
  target_link_libraries(cppvllm PUBLIC ${TORCH_LIBRARIES})
endif()

# ---- programs -----------------------------------------------------------------------------
foreach(app example bench)
  if(EXISTS ${CMAKE_SOURCE_DIR}/${app}.cpp)
    add_executable(${app} ${app}.cpp)
    target_link_libraries(${app} PRIVATE cppvllm)
  endif()
endforeach()

# Your throwaway test file. Delete test.cpp when you are done and this block switches itself off.
# -UNDEBUG keeps assert() alive even in a Release build.
if(EXISTS ${CMAKE_SOURCE_DIR}/test.cpp)
  add_executable(scratch test.cpp)
  target_compile_options(scratch PRIVATE -UNDEBUG)
  target_link_libraries(scratch PRIVATE cppvllm)
endif()