# FindPETSc.cmake
# ----------------
# Find PETSc library
#
# Sets:
#   PETSc_FOUND        - TRUE if PETSc is found
#   PETSC_INCLUDES     - PETSc include directories
#   PETSC_LIBRARIES    - PETSc libraries
#
# Checks PETSC_DIR/PETSC_ARCH environment variables first,
# then falls back to pkg-config.

# Check environment variables
if(DEFINED ENV{PETSC_DIR})
    set(PETSC_DIR "$ENV{PETSC_DIR}")
endif()

if(DEFINED ENV{PETSC_ARCH})
    set(PETSC_ARCH "$ENV{PETSC_ARCH}")
endif()

# Try PETSC_DIR/PETSC_ARCH layout
if(PETSC_DIR)
    set(_petsc_include_dirs "${PETSC_DIR}/include")
    set(_petsc_lib_dirs "${PETSC_DIR}/lib")

    if(PETSC_ARCH)
        list(APPEND _petsc_include_dirs "${PETSC_DIR}/${PETSC_ARCH}/include")
        list(APPEND _petsc_lib_dirs "${PETSC_DIR}/${PETSC_ARCH}/lib")
    endif()

    find_path(PETSC_INCLUDE_DIR petsc.h
        HINTS ${_petsc_include_dirs}
        NO_DEFAULT_PATH
    )

    find_library(PETSC_LIBRARY petsc
        HINTS ${_petsc_lib_dirs}
        NO_DEFAULT_PATH
    )
endif()

# Fallback: pkg-config
if(NOT PETSC_INCLUDE_DIR OR NOT PETSC_LIBRARY)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(_PETSC QUIET PETSc)
        if(_PETSC_FOUND)
            set(PETSC_INCLUDE_DIR ${_PETSC_INCLUDE_DIRS})
            set(PETSC_LIBRARY ${_PETSC_LIBRARIES})
        endif()
    endif()
endif()

# Standard CMake handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
    REQUIRED_VARS PETSC_LIBRARY PETSC_INCLUDE_DIR
)

if(PETSc_FOUND)
    set(PETSC_INCLUDES ${PETSC_INCLUDE_DIR})
    set(PETSC_LIBRARIES ${PETSC_LIBRARY})

    if(NOT TARGET PETSc::PETSc)
        add_library(PETSc::PETSc UNKNOWN IMPORTED)
        set_target_properties(PETSc::PETSc PROPERTIES
            IMPORTED_LOCATION "${PETSC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(PETSC_INCLUDE_DIR PETSC_LIBRARY)
