# GPU Setup Guide for NexusSim

This guide covers GPU acceleration setup for NexusSim using Kokkos.

## Supported GPU Backends

| Backend | Hardware | Compiler | Status |
|---------|----------|----------|--------|
| **CUDA** | NVIDIA GPUs | nvcc | Fully supported |
| **HIP** | AMD GPUs | hipcc | Supported (ROCm 5.0+) |
| **SYCL** | Intel GPUs | dpcpp | Experimental |
| **OpenMP** | Any CPU | gcc/clang | Default fallback |

## Current System Status

```
Execution Space: OpenMP (CPU)
GPU Backend: Not available in WSL2
Peak CPU Performance: 216 million DOFs/sec
```

## WSL2 Limitations

**Important**: WSL2 has limited GPU support:

| Feature | NVIDIA | AMD | Intel |
|---------|--------|-----|-------|
| CUDA/ROCm in WSL2 | ✅ Yes | ❌ No | ⚠️ Limited |
| Native Linux | ✅ Yes | ✅ Yes | ✅ Yes |

WSL2 supports NVIDIA GPU pass-through via CUDA, but **AMD GPU pass-through (ROCm) is not supported**.

## Setup Instructions

### Option 1: NVIDIA GPU (Recommended for WSL2)

```bash
# 1. Install NVIDIA drivers on Windows host
# Download from: https://www.nvidia.com/drivers

# 2. Install CUDA toolkit in WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-2

# 3. Rebuild Kokkos with CUDA
git clone https://github.com/kokkos/kokkos.git
cd kokkos && mkdir build && cd build
cmake .. -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
make -j install

# 4. Rebuild NexusSim
cd /path/to/nexussim/build
cmake .. -DKokkos_DIR=/usr/local/lib/cmake/Kokkos
make -j
```

### Option 2: AMD GPU (Native Linux Only)

```bash
# 1. Install ROCm (requires native Linux, not WSL2)
# See: https://rocm.docs.amd.com/

# For Ubuntu:
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_5.7.0-1_all.deb
sudo apt install ./amdgpu-install_5.7.0-1_all.deb
amdgpu-install --usecase=rocm

# 2. Verify installation
rocminfo
hipcc --version

# 3. Rebuild Kokkos with HIP
git clone https://github.com/kokkos/kokkos.git
cd kokkos && mkdir build && cd build
cmake .. -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_VEGA906=ON
make -j install

# 4. Rebuild NexusSim
cd /path/to/nexussim/build
cmake .. -DKokkos_DIR=/usr/local/lib/cmake/Kokkos
make -j
```

### Option 3: CPU with OpenMP (Current Setup)

This is the current default and works everywhere:

```bash
# Already configured in CMakeLists.txt
cmake .. -DNEXUSSIM_ENABLE_GPU=ON -DNEXUSSIM_ENABLE_OPENMP=ON
make -j

# Set OpenMP environment for best performance
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

## Verifying GPU Setup

Run the Kokkos performance test:

```bash
./bin/kokkos_performance_test
```

Expected output with GPU:
```
Kokkos Configuration:
  Execution Space: Cuda  # or HIP
  Memory Space:    CudaSpace  # or HIPSpace
  GPU Enabled:     Yes
```

Expected output with CPU:
```
Kokkos Configuration:
  Execution Space: OpenMP
  Memory Space:    Host
  GPU Enabled:     No (OpenMP/Serial)
```

## Performance Expectations

| Backend | Elements/sec | DOFs/sec | Notes |
|---------|-------------|----------|-------|
| Serial | ~500K | ~5M | Single thread |
| OpenMP (8 cores) | ~10M | ~200M | Multi-threaded |
| CUDA (RTX 3080) | ~500M | ~2B | GPU accelerated |
| HIP (RX 6800) | ~400M | ~1.5B | GPU accelerated |

## Troubleshooting

### "No CUDA-capable device detected"
- Verify NVIDIA driver is installed on Windows host
- Run `nvidia-smi` in WSL2 to check GPU visibility

### "No ROCm-capable device detected"
- AMD GPUs require native Linux (not WSL2)
- Verify ROCm installation with `rocminfo`
- Check kernel module: `lsmod | grep amdgpu`

### "Kokkos not found"
- Set `Kokkos_DIR` in CMake
- Rebuild Kokkos with desired backend enabled

## Current Benchmark Results

**System**: WSL2 on AMD RX580 (CPU-only mode via OpenMP)

```
============================================================
Peak Performance (OpenMP Backend)
============================================================
  Element Processing: 9.61e+06 elements/sec
  DOF Update Rate:    2.16e+08 DOFs/sec
  Backend:            OpenMP
============================================================
```

For GPU benchmarks, a native Linux installation with proper driver support is required.

---

*Last Updated: 2025-12-29*
