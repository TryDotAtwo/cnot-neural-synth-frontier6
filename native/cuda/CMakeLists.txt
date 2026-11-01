cmake_minimum_required(VERSION 3.24)
project(cnot103_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(cnot103_cuda STATIC src/cnot_cuda.cu)
target_include_directories(cnot103_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cnot103_cuda PUBLIC cxx_std_17)
set_target_properties(cnot103_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cnot103_cuda PUBLIC CUDA::cudart)

install(TARGETS cnot103_cuda
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
