cmake_minimum_required(VERSION 3.24)
project(cnot103_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include(CTest)

add_library(cnot103_cuda STATIC src/cnot_cuda.cu)
target_include_directories(cnot103_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cnot103_cuda PUBLIC cxx_std_17)
set_target_properties(cnot103_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cnot103_cuda PUBLIC CUDA::cudart)

if(BUILD_TESTING)
    add_executable(cnot103_cuda_children_test tests/cnot_cuda_children_test.cu)
    target_link_libraries(cnot103_cuda_children_test PRIVATE cnot103_cuda CUDA::cudart)
    target_compile_features(cnot103_cuda_children_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_children_test PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_children_test COMMAND cnot103_cuda_children_test)
endif()

install(TARGETS cnot103_cuda
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
