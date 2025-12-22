# NexusSim Library Source

include(GNUInstallDirs)

# Collect all source files
set(NEXUSSIM_SOURCES
    # Core infrastructure
    # Note: Currently most of core is header-only

    # FEM Elements
    discretization/fem/solid/hex8.cpp
    discretization/fem/solid/hex20.cpp
    discretization/fem/solid/tet4.cpp
    discretization/fem/solid/tet10.cpp
    discretization/fem/solid/wedge6.cpp
    discretization/fem/shell/shell4.cpp
    discretization/fem/beam/beam2.cpp

    # FEM Solver
    fem/fem_solver.cpp
    fem/contact.cpp

    # I/O
    io/vtk_writer.cpp
    io/mesh_reader.cpp
    io/config_reader.cpp
    io/radioss_reader.cpp

    # Physics
    physics/thermal_solver.cpp

    # Placeholder source file to create library
    nexussim.cpp
)

# Create the library
if(BUILD_SHARED_LIBS)
    add_library(nexussim SHARED ${NEXUSSIM_SOURCES})
else()
    add_library(nexussim STATIC ${NEXUSSIM_SOURCES})
endif()

# Add alias for consistent naming
add_library(NexusSim::nexussim ALIAS nexussim)

# Set target properties
set_target_properties(nexussim PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "nexussim"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(nexussim
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
if(TARGET Eigen3::Eigen)
    target_link_libraries(nexussim PUBLIC Eigen3::Eigen)
endif()

if(TARGET spdlog::spdlog)
    target_link_libraries(nexussim PUBLIC spdlog::spdlog)
endif()

if(TARGET fmt::fmt)
    target_link_libraries(nexussim PUBLIC fmt::fmt)
endif()

# MPI
if(NEXUSSIM_ENABLE_MPI AND MPI_CXX_FOUND)
    target_link_libraries(nexussim PUBLIC MPI::MPI_CXX)
    target_compile_definitions(nexussim PUBLIC NEXUSSIM_ENABLE_MPI)
endif()

# OpenMP
if(NEXUSSIM_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(nexussim PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(nexussim PUBLIC NEXUSSIM_ENABLE_OPENMP)
endif()

# Kokkos/GPU
if(NEXUSSIM_ENABLE_GPU AND TARGET Kokkos::kokkos)
    target_link_libraries(nexussim PUBLIC Kokkos::kokkos)
    target_compile_definitions(nexussim PUBLIC NEXUSSIM_ENABLE_GPU)
endif()

# HDF5
if(TARGET hdf5::hdf5)
    target_link_libraries(nexussim PUBLIC hdf5::hdf5)
endif()

# Precision
if(NEXUSSIM_USE_DOUBLE_PRECISION)
    target_compile_definitions(nexussim PUBLIC NEXUSSIM_REAL=double)
else()
    target_compile_definitions(nexussim PUBLIC NEXUSSIM_REAL=float)
endif()

# Installation
install(TARGETS nexussim
        EXPORT NexusSimTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nexussim
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp"
)

# Export targets
install(EXPORT NexusSimTargets
        FILE NexusSimTargets.cmake
        NAMESPACE NexusSim::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NexusSim
)
