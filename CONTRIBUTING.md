# Contributing to NexusSim

Thank you for your interest in contributing to NexusSim!

## Getting Started

### Prerequisites

- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- CMake 3.20+
- Kokkos 3.7+ (bundled in `external/`)

### Build

```bash
cmake -S . -B build \
  -DNEXUSSIM_ENABLE_MPI=OFF \
  -DNEXUSSIM_BUILD_PYTHON=OFF

cmake --build build -j$(nproc)
```

### Run Tests

```bash
cd build && ctest --output-on-failure -j$(nproc)
```

## Development Workflow

1. Create a feature branch from `master`
2. Make your changes
3. Ensure all tests pass
4. Submit a pull request

## Code Style

- C++20 standard
- 4-space indentation (no tabs)
- ~100 character line limit
- Run `clang-format` before committing: `clang-format -i <file>`
- Configuration in `.clang-format`

## Adding Tests

Tests are standalone executables in `examples/` using `CHECK`/`CHECK_NEAR` macros:

```cpp
#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg << "\n"; } \
} while(0)
```

Add your test target to `CMakeLists.txt`:
```cmake
add_executable(my_test examples/my_test.cpp)
target_link_libraries(my_test PRIVATE nexussim)
add_test(NAME my_test COMMAND my_test)
```

## Coding Conventions

- Use `KOKKOS_INLINE_FUNCTION` for GPU-compatible methods
- Use `nxs::Real` (not `double`) for floating-point values
- Place headers in `include/nexussim/<module>/`
- Use `#pragma once` for include guards
- Namespace hierarchy: `nxs::physics`, `nxs::fem`, `nxs::io`, `nxs::sph`, `nxs::discretization`

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
