set(TARGET_NAME lab04)

file(GLOB_RECURSE TARGET_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE TARGET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories("include")

add_executable(${TARGET_NAME} ${TARGET_HEADERS} ${TARGET_SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE "CL_TARGET_OPENCL_VERSION=120")
target_compile_definitions(${TARGET_NAME} PRIVATE "KERNELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/cl/\"")
target_link_libraries(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX PRIVATE OpenCL::OpenCL)
