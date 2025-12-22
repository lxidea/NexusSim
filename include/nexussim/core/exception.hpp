#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

// Check if source_location is available (not with nvcc)
#if __cplusplus >= 202002L && !defined(__NVCC__) && !defined(__CUDACC__)
    #define NXS_HAS_SOURCE_LOCATION 1
    #include <source_location>
#else
    #define NXS_HAS_SOURCE_LOCATION 0
#endif

namespace nxs {

// ============================================================================
// Source Location Compatibility Layer
// ============================================================================

#if NXS_HAS_SOURCE_LOCATION
using SourceLocation = std::source_location;
#else
// Fallback implementation for environments without std::source_location
struct SourceLocation {
    const char* file_;
    int line_;
    const char* function_;

    constexpr SourceLocation(const char* f = __builtin_FILE(),
                              int l = __builtin_LINE(),
                              const char* fn = __builtin_FUNCTION())
        : file_(f), line_(l), function_(fn) {}

    constexpr const char* file_name() const noexcept { return file_; }
    constexpr int line() const noexcept { return line_; }
    constexpr const char* function_name() const noexcept { return function_; }
    constexpr int column() const noexcept { return 0; }
};
#endif

// ============================================================================
// Exception Base Class
// ============================================================================

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message,
                       const SourceLocation& location = SourceLocation{})
        : std::runtime_error(format_message(message, location))
        , file_(location.file_name())
        , line_(location.line())
        , function_(location.function_name())
    {}

    const char* file() const noexcept { return file_; }
    int line() const noexcept { return line_; }
    const char* function() const noexcept { return function_; }

private:
    const char* file_;
    int line_;
    const char* function_;

    static std::string format_message(const std::string& msg,
                                        const SourceLocation& loc) {
        std::ostringstream oss;
        oss << loc.file_name() << ":"
            << loc.line() << " in "
            << loc.function_name() << "(): "
            << msg;
        return oss.str();
    }
};

// ============================================================================
// Specific Exception Types
// ============================================================================

class LogicError : public Exception {
public:
    explicit LogicError(const std::string& message,
                         const SourceLocation& location = SourceLocation{})
        : Exception(message, location) {}
};

class RuntimeError : public Exception {
public:
    explicit RuntimeError(const std::string& message,
                           const SourceLocation& location = SourceLocation{})
        : Exception(message, location) {}
};

class NotImplementedError : public Exception {
public:
    explicit NotImplementedError(
        const std::string& feature = "This feature",
        const SourceLocation& location = SourceLocation{})
        : Exception(feature + " is not yet implemented", location)
    {}
};

class InvalidArgumentError : public Exception {
public:
    explicit InvalidArgumentError(const std::string& message,
                                   const SourceLocation& location = SourceLocation{})
        : Exception(message, location) {}
};

class OutOfRangeError : public Exception {
public:
    explicit OutOfRangeError(const std::string& message,
                              const SourceLocation& location = SourceLocation{})
        : Exception(message, location) {}
};

class FileIOError : public Exception {
public:
    explicit FileIOError(
        const std::string& filename,
        const std::string& operation = "access",
        const SourceLocation& location = SourceLocation{})
        : Exception("Failed to " + operation + " file: " + filename, location)
    {}
};

class ConvergenceError : public Exception {
public:
    explicit ConvergenceError(
        int iterations,
        double residual,
        const SourceLocation& location = SourceLocation{})
        : Exception(format_convergence_message(iterations, residual), location)
    {}

private:
    static std::string format_convergence_message(int iter, double residual) {
        std::ostringstream oss;
        oss << "Failed to converge after " << iter
            << " iterations (residual = " << residual << ")";
        return oss.str();
    }
};

class GPUError : public Exception {
public:
    explicit GPUError(
        const std::string& operation,
        int error_code = 0,
        const SourceLocation& location = SourceLocation{})
        : Exception(format_gpu_message(operation, error_code), location)
    {}

private:
    static std::string format_gpu_message(const std::string& op, int code) {
        std::ostringstream oss;
        oss << "GPU error during " << op;
        if (code != 0) {
            oss << " (error code: " << code << ")";
        }
        return oss.str();
    }
};

class MPIError : public Exception {
public:
    explicit MPIError(
        const std::string& operation,
        int error_code,
        const SourceLocation& location = SourceLocation{})
        : Exception(format_mpi_message(operation, error_code), location)
    {}

private:
    static std::string format_mpi_message(const std::string& op, int code) {
        std::ostringstream oss;
        oss << "MPI error during " << op << " (error code: " << code << ")";
        return oss.str();
    }
};

// ============================================================================
// Assertion Macros
// ============================================================================

#define NXS_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw ::nxs::LogicError( \
                std::string("Assertion failed: ") + #condition + ": " + (message) \
            ); \
        } \
    } while (false)

#define NXS_REQUIRE(condition, message) \
    do { \
        if (!(condition)) { \
            throw ::nxs::InvalidArgumentError( \
                std::string("Requirement failed: ") + #condition + ": " + (message) \
            ); \
        } \
    } while (false)

#define NXS_CHECK_RANGE(index, size) \
    do { \
        if ((index) >= (size)) { \
            throw ::nxs::OutOfRangeError( \
                "Index " + std::to_string(index) + " out of range [0, " + \
                std::to_string(size) + ")" \
            ); \
        } \
    } while (false)

#define NXS_NOT_IMPLEMENTED() \
    throw ::nxs::NotImplementedError()

#define NXS_NOT_IMPLEMENTED_MSG(msg) \
    throw ::nxs::NotImplementedError(msg)

// In debug mode, enable additional checks
#ifndef NDEBUG
    #define NXS_DEBUG_ASSERT(condition, message) NXS_ASSERT(condition, message)
#else
    #define NXS_DEBUG_ASSERT(condition, message) ((void)0)
#endif

} // namespace nxs
