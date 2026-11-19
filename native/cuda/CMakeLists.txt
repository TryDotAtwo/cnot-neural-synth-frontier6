cmake_minimum_required(VERSION 3.24)
project(cnot103_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
include(CTest)

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

add_library(cnot103_cuda_kernels OBJECT
    src/cnot_cuda.cu
    src/block_histogram.cu
    src/column_diffusion.cu
    src/pair_diffusion.cu
    src/shared_word.cu
    src/basis_gradient.cu
    src/general_compiler.cu
    src/stage26.cu
    src/frequency.cu
    src/shared_relation.cu
    src/bilinear.cu
    src/positive_packet.cu
    src/quadratic.cu
    src/endpoint_response.cu
    src/paid_basis_atlas.cu
    src/paid_shear.cu
    src/paid_basis_features.cu
    src/template_response.cu
    src/guarded_pair.cu
    src/zero_excess.cu
    src/one_excess_structure.cu
    src/one_excess_count.cu
    src/one_excess.cu
    src/matrix_boundary.cu)
target_include_directories(cnot103_cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cnot103_cuda_kernels PUBLIC cxx_std_17)
set_target_properties(cnot103_cuda_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cnot103_cuda_kernels PUBLIC CUDA::cudart)

# Preserve the Rust-facing archive and Torch stream bridge. Standalone kernel
# tests use the same kernel objects without importing unrelated Torch libraries.
add_library(cnot103_cuda STATIC
    $<TARGET_OBJECTS:cnot103_cuda_kernels>
    src/torch_stream.cpp)
target_include_directories(cnot103_cuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_features(cnot103_cuda PUBLIC cxx_std_17)
set_source_files_properties(src/torch_stream.cpp PROPERTIES
    COMPILE_OPTIONS "${TORCH_CXX_FLAGS_LIST}")
set_target_properties(cnot103_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cnot103_cuda PUBLIC CUDA::cudart ${TORCH_LIBRARIES} c10_cuda)

if(BUILD_TESTING)
    add_executable(cnot103_cuda_shared_word_test tests/cnot_cuda_shared_word_test.cu)
    target_link_libraries(cnot103_cuda_shared_word_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_shared_word_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_shared_word_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_shared_word_test COMMAND cnot103_cuda_shared_word_test
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures/shared_word_numpy_2_2_6.bin")
    add_executable(cnot103_cuda_pair_diffusion_test tests/cnot_cuda_pair_diffusion_test.cu)
    target_link_libraries(cnot103_cuda_pair_diffusion_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_pair_diffusion_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_pair_diffusion_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_pair_diffusion_test COMMAND cnot103_cuda_pair_diffusion_test)
    add_test(NAME cnot103_cuda_pair_diffusion_zero_floor COMMAND cnot103_cuda_pair_diffusion_test zero-floor)
    add_executable(cnot103_cuda_column_diffusion_test tests/cnot_cuda_column_diffusion_test.cu)
    target_link_libraries(cnot103_cuda_column_diffusion_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_column_diffusion_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_column_diffusion_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_column_diffusion_test COMMAND cnot103_cuda_column_diffusion_test)
    foreach(mode zero-floor-g01 zero-floor-g02 zero-floor-g03 term-boundary)
        add_test(NAME cnot103_cuda_column_diffusion_${mode} COMMAND cnot103_cuda_column_diffusion_test ${mode})
    endforeach()
    add_executable(cnot103_cuda_children_test tests/cnot_cuda_children_test.cu)
    target_link_libraries(cnot103_cuda_children_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_children_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_children_test PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_children_test COMMAND cnot103_cuda_children_test)

    add_executable(cnot103_cuda_boundary_test tests/cnot_cuda_boundary_test.cu)
    target_link_libraries(cnot103_cuda_boundary_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_boundary_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_boundary_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_boundary_test COMMAND cnot103_cuda_boundary_test)

    add_executable(cnot103_cuda_rank_test tests/cnot_cuda_rank_test.cu)
    target_link_libraries(cnot103_cuda_rank_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_rank_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_rank_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_rank_test COMMAND cnot103_cuda_rank_test)

    add_executable(cnot103_cuda_block_histogram_test tests/cnot_cuda_block_histogram_test.cu)
    target_link_libraries(cnot103_cuda_block_histogram_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_block_histogram_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_block_histogram_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_block_histogram_test COMMAND cnot103_cuda_block_histogram_test)

    add_executable(cnot103_cuda_basis_gradient_test tests/cnot_cuda_basis_gradient_test.cu)
    target_link_libraries(cnot103_cuda_basis_gradient_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_basis_gradient_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_basis_gradient_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_basis_gradient_test COMMAND cnot103_cuda_basis_gradient_test)

    add_executable(cnot103_cuda_general_compiler_test tests/cnot_cuda_general_compiler_test.cu)
    target_link_libraries(cnot103_cuda_general_compiler_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_general_compiler_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_general_compiler_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_general_compiler_test COMMAND cnot103_cuda_general_compiler_test)

    add_executable(cnot103_cuda_stage26_test tests/cnot_cuda_stage26_test.cu)
    target_link_libraries(cnot103_cuda_stage26_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_stage26_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_stage26_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_stage26_test COMMAND cnot103_cuda_stage26_test)
    # Fatal-path checks run in separate processes because a device trap poisons
    # that process's CUDA context. They assert the expected CUDA runtime error.
    add_test(NAME cnot103_cuda_stage26_rejection_test COMMAND cnot103_cuda_stage26_test certificate)
    add_test(NAME cnot103_cuda_stage26_padding_test COMMAND cnot103_cuda_stage26_test padding)

    add_executable(cnot103_cuda_frequency_test tests/cnot_cuda_frequency_test.cu)
    target_link_libraries(cnot103_cuda_frequency_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_frequency_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_frequency_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_frequency_test COMMAND cnot103_cuda_frequency_test)

    add_executable(cnot103_cuda_shared_relation_test tests/cnot_cuda_shared_relation_test.cu)
    target_link_libraries(cnot103_cuda_shared_relation_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_shared_relation_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_shared_relation_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_shared_relation_test COMMAND cnot103_cuda_shared_relation_test)

    add_executable(cnot103_cuda_bilinear_test tests/cnot_cuda_bilinear_test.cu)
    target_link_libraries(cnot103_cuda_bilinear_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_bilinear_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_bilinear_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_bilinear_test COMMAND cnot103_cuda_bilinear_test)

    add_executable(cnot103_cuda_positive_packet_test tests/cnot_cuda_positive_packet_test.cu)
    target_link_libraries(cnot103_cuda_positive_packet_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_positive_packet_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_positive_packet_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_positive_packet_test COMMAND cnot103_cuda_positive_packet_test)

    add_executable(cnot103_cuda_quadratic_test tests/cnot_cuda_quadratic_test.cu)
    target_link_libraries(cnot103_cuda_quadratic_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_quadratic_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_quadratic_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_quadratic_test COMMAND cnot103_cuda_quadratic_test)

    add_executable(cnot103_cuda_endpoint_response_test tests/cnot_cuda_endpoint_response_test.cu)
    target_link_libraries(cnot103_cuda_endpoint_response_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_endpoint_response_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_endpoint_response_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_endpoint_response_test COMMAND cnot103_cuda_endpoint_response_test)

    add_executable(cnot103_cuda_paid_basis_atlas_test tests/cnot_cuda_paid_basis_atlas_test.cu)
    target_link_libraries(cnot103_cuda_paid_basis_atlas_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_paid_basis_atlas_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_paid_basis_atlas_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_paid_basis_atlas_test COMMAND cnot103_cuda_paid_basis_atlas_test)

    add_executable(cnot103_cuda_paid_basis_test tests/cnot_cuda_paid_basis_test.cu)
    target_link_libraries(cnot103_cuda_paid_basis_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_paid_basis_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_paid_basis_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_paid_basis_test COMMAND cnot103_cuda_paid_basis_test)

    add_executable(cnot103_cuda_template_response_test tests/cnot_cuda_template_response_test.cu)
    target_link_libraries(cnot103_cuda_template_response_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_template_response_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_template_response_test PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_template_response_test COMMAND cnot103_cuda_template_response_test)
    add_executable(cnot103_cuda_guarded_pair_test tests/cnot_cuda_guarded_pair_test.cu)
    target_link_libraries(cnot103_cuda_guarded_pair_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_guarded_pair_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_guarded_pair_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_guarded_pair_test COMMAND cnot103_cuda_guarded_pair_test)
    add_test(NAME cnot103_cuda_guarded_pair_rejection_test COMMAND cnot103_cuda_guarded_pair_test certificate)
    add_test(NAME cnot103_cuda_guarded_pair_padding_test COMMAND cnot103_cuda_guarded_pair_test padding)
    add_test(NAME cnot103_cuda_guarded_pair_singular_test COMMAND cnot103_cuda_guarded_pair_test singular)
    add_executable(cnot103_cuda_zero_excess_test tests/cnot_cuda_zero_excess_test.cu)
    target_link_libraries(cnot103_cuda_zero_excess_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_zero_excess_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_zero_excess_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_zero_excess_test COMMAND cnot103_cuda_zero_excess_test)
    add_test(NAME cnot103_cuda_zero_excess_exact_rejection_test COMMAND cnot103_cuda_zero_excess_test exact)
    add_test(NAME cnot103_cuda_zero_excess_negative_rejection_test COMMAND cnot103_cuda_zero_excess_test negative)
    add_test(NAME cnot103_cuda_zero_excess_cycle_test COMMAND cnot103_cuda_zero_excess_test cycle)
    add_test(NAME cnot103_cuda_zero_excess_padding_test COMMAND cnot103_cuda_zero_excess_test padding)
    add_executable(cnot103_cuda_one_excess_math_test tests/cnot_cuda_one_excess_math_test.cu)
    target_link_libraries(cnot103_cuda_one_excess_math_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_one_excess_math_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_one_excess_math_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_one_excess_math_test COMMAND cnot103_cuda_one_excess_math_test)
    add_test(NAME cnot103_cuda_one_excess_math_overflow_test COMMAND cnot103_cuda_one_excess_math_test overflow)
    add_test(NAME cnot103_cuda_one_excess_math_underflow_test COMMAND cnot103_cuda_one_excess_math_test underflow)
    add_test(NAME cnot103_cuda_one_excess_math_division_test COMMAND cnot103_cuda_one_excess_math_test division)
    add_executable(cnot103_cuda_one_excess_structure_test tests/cnot_cuda_one_excess_structure_test.cu)
    target_link_libraries(cnot103_cuda_one_excess_structure_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_one_excess_structure_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_one_excess_structure_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_one_excess_structure_test COMMAND cnot103_cuda_one_excess_structure_test)
    foreach(mode length zero repeat padding singular final double protected unused)
        add_test(NAME cnot103_cuda_one_excess_structure_${mode}_test COMMAND cnot103_cuda_one_excess_structure_test ${mode})
    endforeach()
    add_executable(cnot103_cuda_one_excess_count_test tests/cnot_cuda_one_excess_count_test.cu)
    target_link_libraries(cnot103_cuda_one_excess_count_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_one_excess_count_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_one_excess_count_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_one_excess_count_test COMMAND cnot103_cuda_one_excess_count_test)
    foreach(mode length zero repeat padding singular final double protected unused)
        add_test(NAME cnot103_cuda_one_excess_count_${mode}_test COMMAND cnot103_cuda_one_excess_count_test ${mode})
    endforeach()
    add_executable(cnot103_cuda_one_excess_test tests/cnot_cuda_one_excess_test.cu)
    target_link_libraries(cnot103_cuda_one_excess_test PRIVATE cnot103_cuda_kernels)
    target_compile_features(cnot103_cuda_one_excess_test PRIVATE cxx_std_17)
    set_target_properties(cnot103_cuda_one_excess_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME cnot103_cuda_one_excess_test COMMAND cnot103_cuda_one_excess_test)
    foreach(mode class changed templates bag transport distance count high_count gate unused length prerequisite prereq_count prereq_status padding singular publish)
        add_test(NAME cnot103_cuda_one_excess_classifier_${mode}_test COMMAND cnot103_cuda_one_excess_test ${mode})
    endforeach()
endif()

install(TARGETS cnot103_cuda
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
