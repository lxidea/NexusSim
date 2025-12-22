from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps

class NexusSimConan(ConanFile):
    name = "nexussim"
    version = "0.1.0"
    license = "Apache-2.0"
    author = "NexusSim Development Team"
    url = "https://github.com/nexussim/nexussim"
    description = "Next-Generation Computational Mechanics Framework"
    topics = ("fem", "cfd", "hpc", "gpu", "multi-physics")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "enable_mpi": [True, False],
        "enable_gpu": [True, False],
        "enable_openmp": [True, False],
        "enable_tests": [True, False],
        "enable_python": [True, False],
        "double_precision": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "enable_mpi": True,
        "enable_gpu": True,
        "enable_openmp": True,
        "enable_tests": True,
        "enable_python": True,
        "double_precision": True,
    }

    exports_sources = "CMakeLists.txt", "src/*", "include/*", "cmake/*", "python/*"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def requirements(self):
        # Core dependencies
        self.requires("eigen/3.4.0")
        self.requires("spdlog/1.12.0")
        self.requires("fmt/10.1.1")

        # HDF5 for I/O
        self.requires("hdf5/1.14.0")

        # Optional: MPI (system-provided usually)
        # if self.options.enable_mpi:
        #     self.requires("openmpi/4.1.5")  # Usually system-provided

        # Kokkos for GPU (may need custom build)
        if self.options.enable_gpu:
            # self.requires("kokkos/4.1.00")  # May need custom build with GPU backends
            pass

        # Testing
        if self.options.enable_tests:
            self.requires("catch2/3.4.0")
            self.requires("benchmark/1.8.3")

        # Python bindings
        if self.options.enable_python:
            self.requires("pybind11/2.11.1")

    def build_requirements(self):
        self.tool_requires("cmake/3.27.7")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        # Pass options to CMake
        tc.variables["NEXUSSIM_ENABLE_MPI"] = self.options.enable_mpi
        tc.variables["NEXUSSIM_ENABLE_GPU"] = self.options.enable_gpu
        tc.variables["NEXUSSIM_ENABLE_OPENMP"] = self.options.enable_openmp
        tc.variables["NEXUSSIM_BUILD_TESTS"] = self.options.enable_tests
        tc.variables["NEXUSSIM_BUILD_PYTHON"] = self.options.enable_python
        tc.variables["NEXUSSIM_USE_DOUBLE_PRECISION"] = self.options.double_precision
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["nexussim"]
        self.cpp_info.includedirs = ["include"]
