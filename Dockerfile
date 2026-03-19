# NexusSim Development Container
# Multi-stage build: build + test, then slim runtime

# Stage 1: Build and test
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake g++-13 gcc-13 \
    libkokkos-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 13 as default
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100

WORKDIR /nexussim
COPY . .

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEXUSSIM_ENABLE_MPI=OFF \
    -DNEXUSSIM_BUILD_PYTHON=OFF \
    -DNEXUSSIM_ENABLE_GPU=OFF \
    && cmake --build build -j$(nproc)

RUN cd build && ctest --output-on-failure -j$(nproc) -L '!known_fail'

# Stage 2: Runtime (headers + library only)
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /nexussim/include /usr/local/include
COPY --from=builder /nexussim/build/lib /usr/local/lib

WORKDIR /workspace
