cmake_minimum_required(VERSION 3.22)
project(pocketworkbench LANGUAGES C CXX)

# Inference code MUST be optimized: AGP passes CMAKE_BUILD_TYPE=Debug for debug
# builds, which compiles llama.cpp/whisper.cpp at -O0 (~50-100x slower).
# Force a Release toolchain regardless of the AGP build type.
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
message(STATUS "Native build type forced to: ${CMAKE_BUILD_TYPE}")

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/llama.cpp llama_build)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/whisper.cpp whisper_build)
add_library(pocketnative SHARED bridge.cpp)
target_include_directories(pocketnative PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/llama.cpp/include ${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/llama.cpp/ggml/include ${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/whisper.cpp/include)
target_link_libraries(pocketnative PRIVATE llama whisper log android)

# Snapdragon 8-class SoCs are ARMv8.2+ (dotprod + i8mm + fp16): enable those
# extensions for the CPU inference kernels (ggml-cpu aarch64 gemv/gemm paths).
# This is what GGML_NATIVE=OFF disabled. Requirements: ARMv8.2-a device or newer
# (all flagship SoCs since 2019). On older ARMv8.0 devices generation would
# SIGILL; remove the -march flag below to restore universal compatibility.
set(AARCH64_FLAGS -O3 -march=armv8.2-a+dotprod+i8mm+fp16)
foreach(t ggml-cpu llama whisper pocketnative)
    if(TARGET ${t})
        target_compile_options(${t} PRIVATE ${AARCH64_FLAGS})
        message(STATUS "Applied optimized flags to target: ${t}")
    endif()
endforeach()
