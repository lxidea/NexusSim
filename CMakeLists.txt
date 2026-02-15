cmake_minimum_required(VERSION 3.20)

project(NexusSim
    VERSION 0.1.0
    DESCRIPTION "Next-Generation Computational Mechanics Framework"
    LANGUAGES CXX C
)

# Project-wide C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(NEXUSSIM_BUILD_TESTS "Build test suite" ON)
option(NEXUSSIM_BUILD_PYTHON "Build Python bindings" ON)
option(NEXUSSIM_BUILD_EXAMPLES "Build example programs" ON)
option(NEXUSSIM_ENABLE_GPU "Enable GPU acceleration via Kokkos" ON)
option(NEXUSSIM_ENABLE_MPI "Enable MPI parallelization" ON)
option(NEXUSSIM_ENABLE_OPENMP "Enable OpenMP threading" ON)
option(NEXUSSIM_USE_DOUBLE_PRECISION "Use double precision (off=single)" ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Module path for custom CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Compiler flags
# Note: Simplified for Kokkos nvcc_wrapper compatibility
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        add_compile_options(-Wall -Wextra -O0)
    else()
        add_compile_options(-Wall -O2)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        add_compile_options(/W4 /Od)
    else()
        add_compile_options(/W4 /O2)
    endif()
endif()

# Find dependencies
find_package(Threads REQUIRED)

# MPI - Check custom location first
if(NEXUSSIM_ENABLE_MPI)
    # Add custom MPICH path
    set(MPI_HOME "/opt/mpich-4.3.0b1" CACHE PATH "MPICH installation directory")
    list(APPEND CMAKE_PREFIX_PATH ${MPI_HOME})

    find_package(MPI REQUIRED)
    add_compile_definitions(NEXUSSIM_HAVE_MPI)

    message(STATUS "MPI found:")
    message(STATUS "  MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
    message(STATUS "  MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
endif()

# OpenMP
if(NEXUSSIM_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_compile_definitions(NEXUSSIM_HAVE_OPENMP)
endif()

# Kokkos for GPU portability - Check custom location first
if(NEXUSSIM_ENABLE_GPU)
    # Add custom Kokkos path
    set(Kokkos_DIR "/usr/local/lib/cmake/Kokkos" CACHE PATH "Kokkos CMake directory")
    list(APPEND CMAKE_PREFIX_PATH /usr/local)

    # Find CUDAToolkit BEFORE Kokkos to define CUDA::cuda_driver and CUDA::cudart targets
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.6" CACHE PATH "CUDA Toolkit directory")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")
        message(STATUS "  CUDA_TOOLKIT_ROOT_DIR: ${CUDAToolkit_ROOT}")
    endif()

    find_package(Kokkos REQUIRED)
    add_compile_definitions(NEXUSSIM_HAVE_KOKKOS)

    message(STATUS "Kokkos found:")
    message(STATUS "  Kokkos_VERSION: ${Kokkos_VERSION}")
    message(STATUS "  Kokkos_CXX_COMPILER: ${Kokkos_CXX_COMPILER}")
    message(STATUS "  Kokkos_DEVICES: ${Kokkos_DEVICES}")
endif()

# HDF5 for I/O (optional for now)
find_package(HDF5 COMPONENTS C CXX)
if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_VERSION}")
    add_compile_definitions(NEXUSSIM_HAVE_HDF5)
else()
    message(WARNING "HDF5 not found - I/O features will be limited")
endif()

# yaml-cpp for enhanced YAML parsing
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp found: ${yaml-cpp_VERSION}")
    add_compile_definitions(NEXUSSIM_HAVE_YAML_CPP)
else()
    message(STATUS "yaml-cpp not found - using custom YAML parser")
endif()

# Eigen for linear algebra (optional for now)
find_package(Eigen3)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
else()
    message(WARNING "Eigen3 not found - linear algebra features will be limited")
endif()

# spdlog for logging
find_package(spdlog)
if(NOT spdlog_FOUND)
    message(WARNING "spdlog not found - using header-only mode")
endif()

# Precision setting
if(NEXUSSIM_USE_DOUBLE_PRECISION)
    add_compile_definitions(NEXUSSIM_REAL=double)
else()
    add_compile_definitions(NEXUSSIM_REAL=float)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include  # For generated headers
)

# Subdirectories
add_subdirectory(src)

if(NEXUSSIM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(src/tests)
endif()

if(NEXUSSIM_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(NEXUSSIM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT NexusSimTargets
    FILE NexusSimTargets.cmake
    NAMESPACE NexusSim::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NexusSim
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NexusSimConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/NexusSimConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NexusSimConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NexusSim
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NexusSimConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NexusSimConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NexusSim
)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "NexusSim Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  MPI enabled: ${NEXUSSIM_ENABLE_MPI}")
message(STATUS "  GPU enabled: ${NEXUSSIM_ENABLE_GPU}")
message(STATUS "  OpenMP enabled: ${NEXUSSIM_ENABLE_OPENMP}")
message(STATUS "  Double precision: ${NEXUSSIM_USE_DOUBLE_PRECISION}")
message(STATUS "  Build tests: ${NEXUSSIM_BUILD_TESTS}")
message(STATUS "  Build Python: ${NEXUSSIM_BUILD_PYTHON}")
message(STATUS "========================================")
add_executable(hex20_force_direction_test examples/hex20_force_direction_test.cpp)
target_link_libraries(hex20_force_direction_test PRIVATE nexussim)
add_executable(hex8_force_direction_test examples/hex8_force_direction_test.cpp)
target_link_libraries(hex8_force_direction_test PRIVATE nexussim)
add_executable(hex20_mass_matrix_check examples/hex20_mass_matrix_check.cpp)
target_link_libraries(hex20_mass_matrix_check PRIVATE nexussim)
add_executable(hex20_static_load_test examples/hex20_static_load_test.cpp)
target_link_libraries(hex20_static_load_test PRIVATE nexussim)
add_executable(hex20_single_step_debug examples/hex20_single_step_debug.cpp)
target_link_libraries(hex20_single_step_debug PRIVATE nexussim)
add_executable(hex20_multistep_debug examples/hex20_multistep_debug.cpp)
target_link_libraries(hex20_multistep_debug PRIVATE nexussim)
add_executable(hex20_debug_test examples/hex20_debug_test.cpp)
target_link_libraries(hex20_debug_test PRIVATE nexussim)
add_executable(vonmises_plasticity_test examples/vonmises_plasticity_test.cpp)
target_link_libraries(vonmises_plasticity_test PRIVATE nexussim)
add_executable(johnson_cook_test examples/johnson_cook_test.cpp)
target_link_libraries(johnson_cook_test PRIVATE nexussim)
add_executable(neohookean_test examples/neohookean_test.cpp)
target_link_libraries(neohookean_test PRIVATE nexussim)
add_executable(implicit_newmark_test examples/implicit_newmark_test.cpp)
target_link_libraries(implicit_newmark_test PRIVATE nexussim)
add_executable(contact_test examples/contact_test.cpp)
target_link_libraries(contact_test PRIVATE nexussim)
add_executable(vtk_animation_demo examples/vtk_animation_demo.cpp)
target_link_libraries(vtk_animation_demo PRIVATE nexussim)
add_executable(adaptive_timestep_test examples/adaptive_timestep_test.cpp)
target_link_libraries(adaptive_timestep_test PRIVATE nexussim)
add_executable(large_deformation_test examples/large_deformation_test.cpp)
target_link_libraries(large_deformation_test PRIVATE nexussim)
add_executable(element_erosion_test examples/element_erosion_test.cpp)
target_link_libraries(element_erosion_test PRIVATE nexussim)
add_executable(sparse_solver_test examples/sparse_solver_test.cpp)
target_link_libraries(sparse_solver_test PRIVATE nexussim)
add_executable(gpu_sparse_test examples/gpu_sparse_test.cpp)
target_link_libraries(gpu_sparse_test PRIVATE nexussim)
add_executable(node_to_surface_contact_test examples/node_to_surface_contact_test.cpp)
target_link_libraries(node_to_surface_contact_test PRIVATE nexussim)
add_executable(mesh_partition_test examples/mesh_partition_test.cpp)
target_link_libraries(mesh_partition_test PRIVATE nexussim)
add_executable(hex20_derivative_verify examples/hex20_derivative_verify.cpp)
target_link_libraries(hex20_derivative_verify PRIVATE nexussim)
add_executable(hex20_force_compare examples/hex20_force_compare.cpp)
target_link_libraries(hex20_force_compare PRIVATE nexussim)
add_executable(hex20_dynamic_debug examples/hex20_dynamic_debug.cpp)
target_link_libraries(hex20_dynamic_debug PRIVATE nexussim)
add_executable(hex20_stiffness_check examples/hex20_stiffness_check.cpp)
target_link_libraries(hex20_stiffness_check PRIVATE nexussim)

# New element tests (Phase 3A)
add_executable(shell3_element_test examples/shell3_element_test.cpp)
target_link_libraries(shell3_element_test PRIVATE nexussim)
add_executable(truss_element_test examples/truss_element_test.cpp)
target_link_libraries(truss_element_test PRIVATE nexussim)
add_executable(spring_damper_test examples/spring_damper_test.cpp)
target_link_libraries(spring_damper_test PRIVATE nexussim)
add_executable(radioss_reader_test examples/radioss_reader_test.cpp)
target_link_libraries(radioss_reader_test PRIVATE nexussim)

# Phase 3B tests
add_executable(thermal_coupling_test examples/thermal_coupling_test.cpp)
target_link_libraries(thermal_coupling_test PRIVATE nexussim)

add_executable(time_integration_test examples/time_integration_test.cpp)
target_link_libraries(time_integration_test PRIVATE nexussim)

# Phase 3C tests (SPH)
add_executable(sph_solver_test examples/sph_solver_test.cpp)
target_link_libraries(sph_solver_test PRIVATE nexussim)

add_executable(fem_sph_coupling_test examples/fem_sph_coupling_test.cpp)
target_link_libraries(fem_sph_coupling_test PRIVATE nexussim)

# Wave 3 tests (Implicit Solver)
add_executable(implicit_solver_test examples/implicit_solver_test.cpp)
target_link_libraries(implicit_solver_test PRIVATE nexussim)

# Wave 3 Phase 2: FEM Static Solver
add_executable(fem_static_test examples/fem_static_test.cpp)
target_link_libraries(fem_static_test PRIVATE nexussim)

# Wave 3 Phase 2: FEM Implicit Dynamic Solver
add_executable(implicit_dynamic_test examples/implicit_dynamic_test.cpp)
target_link_libraries(implicit_dynamic_test PRIVATE nexussim)

# Wave 4: Peridynamics Integration
add_executable(pd_bar_tension_test examples/pd_bar_tension_test.cpp)
target_link_libraries(pd_bar_tension_test PRIVATE nexussim)

add_executable(pd_validation_test examples/pd_validation_test.cpp)
target_link_libraries(pd_validation_test PRIVATE nexussim)

add_executable(pd_fem_coupling_test examples/pd_fem_coupling_test.cpp)
target_link_libraries(pd_fem_coupling_test PRIVATE nexussim)

add_executable(fem_pd_integration_test examples/fem_pd_integration_test.cpp)
target_link_libraries(fem_pd_integration_test PRIVATE nexussim)

add_executable(mpi_partition_test examples/mpi_partition_test.cpp)
target_link_libraries(mpi_partition_test PRIVATE nexussim)

# Wave 5: Performance Optimization
add_executable(comprehensive_benchmark examples/comprehensive_benchmark.cpp)
target_link_libraries(comprehensive_benchmark PRIVATE nexussim)

# Wave 1: Material Models
add_executable(material_models_test examples/material_models_test.cpp)
target_link_libraries(material_models_test PRIVATE nexussim)

# Wave 2: Failure/Damage Models
add_executable(failure_models_test examples/failure_models_test.cpp)
target_link_libraries(failure_models_test PRIVATE nexussim)

# Wave 3: Rigid Bodies, Constraints, Rigid Walls
add_executable(rigid_body_test examples/rigid_body_test.cpp)
target_link_libraries(rigid_body_test PRIVATE nexussim)

# Wave 4: Loads System
add_executable(loads_system_test examples/loads_system_test.cpp)
target_link_libraries(loads_system_test PRIVATE nexussim)

# Wave 5: Tied Contact + EOS
add_executable(tied_contact_eos_test examples/tied_contact_eos_test.cpp)
target_link_libraries(tied_contact_eos_test PRIVATE nexussim)

add_executable(kokkos_performance_test examples/kokkos_performance_test.cpp)
target_link_libraries(kokkos_performance_test PRIVATE Kokkos::kokkos)

# Wave 6: Restart/Checkpoint + Output
add_executable(restart_output_test examples/restart_output_test.cpp)
target_link_libraries(restart_output_test PRIVATE nexussim)

# Wave 7: Composite Ply Stacking
add_executable(composite_layup_test examples/composite_layup_test.cpp)
target_link_libraries(composite_layup_test PRIVATE nexussim)

# Wave 8: Sensors, Controls, and ALE
add_executable(sensor_ale_test examples/sensor_ale_test.cpp)
target_link_libraries(sensor_ale_test PRIVATE nexussim)

# Realistic multi-system integration tests
add_executable(realistic_crash_test examples/realistic_crash_test.cpp)
target_link_libraries(realistic_crash_test PRIVATE nexussim)

# Hertzian Contact + Mortar Contact Tests
add_executable(hertzian_mortar_test examples/hertzian_mortar_test.cpp)
target_link_libraries(hertzian_mortar_test PRIVATE nexussim)

# Wave 6 Enhanced: Extended Checkpoint + Output
add_executable(enhanced_output_test examples/enhanced_output_test.cpp)
target_link_libraries(enhanced_output_test PRIVATE nexussim)

# LS-DYNA Reader Extension Tests
add_executable(lsdyna_reader_ext_test examples/lsdyna_reader_ext_test.cpp)
target_link_libraries(lsdyna_reader_ext_test PRIVATE nexussim)
