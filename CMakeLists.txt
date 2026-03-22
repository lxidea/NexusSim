cmake_minimum_required(VERSION 3.20)

project(NexusSim
    VERSION 1.0.0
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

# MPI
if(NEXUSSIM_ENABLE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(NEXUSSIM_HAVE_MPI)

    message(STATUS "MPI found:")
    message(STATUS "  MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
    message(STATUS "  MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "  MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
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

# PETSc solver (optional)
option(NEXUSSIM_ENABLE_PETSC "Enable PETSc solver" OFF)
if(NEXUSSIM_ENABLE_PETSC)
    find_package(PETSc)
    if(PETSc_FOUND)
        add_compile_definitions(NEXUSSIM_HAVE_PETSC)
        message(STATUS "PETSc found: ${PETSC_LIBRARIES}")
    else()
        message(WARNING "PETSc not found - PETSc solver will be disabled")
    endif()
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

# Test targets (extracted to cmake/TestTargets.cmake for maintainability)
include(cmake/TestTargets.cmake)
