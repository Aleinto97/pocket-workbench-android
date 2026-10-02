cmake_minimum_required(VERSION 3.22)
project(pocketworkbench LANGUAGES C CXX)

# Speech code MUST be optimized: AGP passes CMAKE_BUILD_TYPE=Debug for debug
# builds, which compiles whisper.cpp at -O0 (~50-100x slower).
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
message(STATUS "Native build type forced to: ${CMAKE_BUILD_TYPE}")

# The LLM inference engine is pure Rust (rust/pocketinfer -> libpocketinfer.so),
# built by scripts/build-rust.sh and staged into app/build/rustjni/arm64-v8a.
# This CMake project only builds whisper.cpp for on-device speech recognition.
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/whisper.cpp whisper_build)
add_library(pocketnative SHARED speech_bridge.cpp)
target_include_directories(pocketnative PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/whisper.cpp/include)
target_link_libraries(pocketnative PRIVATE whisper log android)

# Snapdragon 8-class SoCs are ARMv8.2+ (dotprod + i8mm + fp16): enable those
# extensions for the whisper CPU kernels. On older ARMv8.0 devices generation
# would SIGILL; remove the -march flag below to restore compatibility.
set(AARCH64_FLAGS -O3 -march=armv8.2-a+dotprod+i8mm+fp16)
foreach(t whisper pocketnative)
    if(TARGET ${t})
        target_compile_options(${t} PRIVATE ${AARCH64_FLAGS})
        message(STATUS "Applied optimized flags to target: ${t}")
    endif()
endforeach()
