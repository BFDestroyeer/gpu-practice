cmake_minimum_required(VERSION 3.19)

project(lab01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL 1.2 REQUIRED)

set(TARGET_NAME lab01)

file(GLOB_RECURSE TARGET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(${TARGET_NAME} ${TARGET_SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE "CL_TARGET_OPENCL_VERSION=220")
target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)