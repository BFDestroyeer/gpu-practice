cmake_minimum_required(VERSION 3.19)

project(lab01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL 1.2 REQUIRED)

set(TARGET_NAME lab01)

file(GLOB_RECURSE TARGET_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE TARGET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories("include")

add_executable(${TARGET_NAME} ${TARGET_HEADERS} ${TARGET_SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE "CL_TARGET_OPENCL_VERSION=120")
target_compile_definitions(${TARGET_NAME} PRIVATE "KERNELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/cl/\"")
target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)