#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/exception.hpp>
#include <nexussim/core/logger.hpp>

#ifdef NEXUSSIM_HAVE_MPI
#include <mpi.h>
#endif

namespace nxs {

// ============================================================================
// MPI Communicator Wrapper
// ============================================================================

#ifdef NEXUSSIM_HAVE_MPI

class MPIManager {
public:
    static MPIManager& instance() {
        static MPIManager manager;
        return manager;
    }

    void initialize(int* argc = nullptr, char*** argv = nullptr) {
        if (is_initialized_) {
            NXS_LOG_WARN("MPI already initialized");
            return;
        }

        int provided;
        int requested = MPI_THREAD_FUNNELED;

        int err = MPI_Init_thread(argc, argv, requested, &provided);
        if (err != MPI_SUCCESS) {
            throw MPIError("MPI_Init_thread", err);
        }

        is_initialized_ = true;

        // Get rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        if (rank_ == 0) {
            print_configuration(provided);
        }
    }

    void finalize() {
        if (!is_initialized_) {
            return;
        }

        MPI_Finalize();
        is_initialized_ = false;

        if (rank_ == 0) {
            NXS_LOG_INFO("MPI finalized");
        }
    }

    bool is_initialized() const { return is_initialized_; }

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool is_root() const { return rank_ == 0; }

    MPI_Comm comm_world() const { return MPI_COMM_WORLD; }

    void barrier(MPI_Comm comm = MPI_COMM_WORLD) const {
        MPI_Barrier(comm);
    }

    void print_configuration(int thread_level) const {
        NXS_LOG_INFO("MPI Configuration:");
        NXS_LOG_INFO("  Number of processes: {}", size_);
        NXS_LOG_INFO("  Current rank: {}", rank_);

        const char* thread_support;
        switch (thread_level) {
            case MPI_THREAD_SINGLE:
                thread_support = "MPI_THREAD_SINGLE";
                break;
            case MPI_THREAD_FUNNELED:
                thread_support = "MPI_THREAD_FUNNELED";
                break;
            case MPI_THREAD_SERIALIZED:
                thread_support = "MPI_THREAD_SERIALIZED";
                break;
            case MPI_THREAD_MULTIPLE:
                thread_support = "MPI_THREAD_MULTIPLE";
                break;
            default:
                thread_support = "Unknown";
        }
        NXS_LOG_INFO("  Thread support: {}", thread_support);

        // Get MPI version
        int version, subversion;
        MPI_Get_version(&version, &subversion);
        NXS_LOG_INFO("  MPI version: {}.{}", version, subversion);

        // Get library version string
        char lib_version[MPI_MAX_LIBRARY_VERSION_STRING];
        int len;
        MPI_Get_library_version(lib_version, &len);
        NXS_LOG_INFO("  MPI library: {}", std::string(lib_version, len));
    }

private:
    MPIManager() = default;

    ~MPIManager() {
        if (is_initialized_) {
            finalize();
        }
    }

    // Delete copy/move
    MPIManager(const MPIManager&) = delete;
    MPIManager& operator=(const MPIManager&) = delete;

    bool is_initialized_ = false;
    int rank_ = 0;
    int size_ = 1;
};

// ============================================================================
// MPI Collective Operations
// ============================================================================

// Broadcast
template<typename T>
inline void broadcast(T* buffer, int count, int root, MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Datatype dtype;

    if constexpr (std::is_same_v<T, int>) {
        dtype = MPI_INT;
    } else if constexpr (std::is_same_v<T, long>) {
        dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        dtype = MPI_DOUBLE;
    } else {
        dtype = MPI_BYTE;
        count *= sizeof(T);
    }

    int err = MPI_Bcast(buffer, count, dtype, root, comm);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Bcast", err);
    }
}

// Reduce (sum)
template<typename T>
inline void reduce_sum(const T* sendbuf, T* recvbuf, int count, int root,
                       MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Datatype dtype;

    if constexpr (std::is_same_v<T, int>) {
        dtype = MPI_INT;
    } else if constexpr (std::is_same_v<T, long>) {
        dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        dtype = MPI_DOUBLE;
    } else {
        static_assert(std::is_arithmetic_v<T>, "Unsupported type for MPI reduce");
    }

    int err = MPI_Reduce(sendbuf, recvbuf, count, dtype, MPI_SUM, root, comm);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Reduce", err);
    }
}

// All-reduce (sum)
template<typename T>
inline void allreduce_sum(const T* sendbuf, T* recvbuf, int count,
                          MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Datatype dtype;

    if constexpr (std::is_same_v<T, int>) {
        dtype = MPI_INT;
    } else if constexpr (std::is_same_v<T, long>) {
        dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        dtype = MPI_DOUBLE;
    } else {
        static_assert(std::is_arithmetic_v<T>, "Unsupported type for MPI allreduce");
    }

    int err = MPI_Allreduce(sendbuf, recvbuf, count, dtype, MPI_SUM, comm);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Allreduce", err);
    }
}

// All-reduce (max)
template<typename T>
inline void allreduce_max(const T* sendbuf, T* recvbuf, int count,
                          MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Datatype dtype;

    if constexpr (std::is_same_v<T, int>) {
        dtype = MPI_INT;
    } else if constexpr (std::is_same_v<T, long>) {
        dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        dtype = MPI_DOUBLE;
    } else {
        static_assert(std::is_arithmetic_v<T>, "Unsupported type for MPI allreduce");
    }

    int err = MPI_Allreduce(sendbuf, recvbuf, count, dtype, MPI_MAX, comm);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Allreduce", err);
    }
}

// All-reduce (min)
template<typename T>
inline void allreduce_min(const T* sendbuf, T* recvbuf, int count,
                          MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Datatype dtype;

    if constexpr (std::is_same_v<T, int>) {
        dtype = MPI_INT;
    } else if constexpr (std::is_same_v<T, long>) {
        dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        dtype = MPI_DOUBLE;
    } else {
        static_assert(std::is_arithmetic_v<T>, "Unsupported type for MPI allreduce");
    }

    int err = MPI_Allreduce(sendbuf, recvbuf, count, dtype, MPI_MIN, comm);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Allreduce", err);
    }
}

#else // !NEXUSSIM_HAVE_MPI

// Dummy MPI manager when MPI is not available
class MPIManager {
public:
    static MPIManager& instance() {
        static MPIManager manager;
        return manager;
    }

    void initialize(int* = nullptr, char*** = nullptr) {
        NXS_LOG_WARN("MPI support not enabled - running in serial mode");
    }

    void finalize() {}

    bool is_initialized() const { return false; }
    int rank() const { return 0; }
    int size() const { return 1; }
    bool is_root() const { return true; }

    void barrier() const {}

    void print_configuration(int = 0) const {
        NXS_LOG_INFO("MPI: disabled (serial mode)");
    }
};

// Dummy collective operations
template<typename T>
inline void broadcast(T*, int, int) {}

template<typename T>
inline void reduce_sum(const T*, T*, int, int) {}

template<typename T>
inline void allreduce_sum(const T* sendbuf, T* recvbuf, int count) {
    // In serial mode, just copy
    for (int i = 0; i < count; ++i) {
        recvbuf[i] = sendbuf[i];
    }
}

template<typename T>
inline void allreduce_max(const T* sendbuf, T* recvbuf, int count) {
    for (int i = 0; i < count; ++i) {
        recvbuf[i] = sendbuf[i];
    }
}

template<typename T>
inline void allreduce_min(const T* sendbuf, T* recvbuf, int count) {
    for (int i = 0; i < count; ++i) {
        recvbuf[i] = sendbuf[i];
    }
}

#endif // NEXUSSIM_HAVE_MPI

} // namespace nxs
