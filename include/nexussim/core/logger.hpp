#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>

#ifdef NEXUSSIM_HAVE_SPDLOG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#endif

namespace nxs {

// ============================================================================
// Logger Class
// ============================================================================

class Logger {
public:
    enum class Level {
#ifdef NEXUSSIM_HAVE_SPDLOG
        Trace = SPDLOG_LEVEL_TRACE,
        Debug = SPDLOG_LEVEL_DEBUG,
        Info = SPDLOG_LEVEL_INFO,
        Warn = SPDLOG_LEVEL_WARN,
        Error = SPDLOG_LEVEL_ERROR,
        Critical = SPDLOG_LEVEL_CRITICAL,
        Off = SPDLOG_LEVEL_OFF
#else
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warn = 3,
        Error = 4,
        Critical = 5,
        Off = 6
#endif
    };

    // Get the global logger instance (singleton)
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    // Initialize logger with console output
    void init_console(Level level = Level::Info) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(static_cast<spdlog::level::level_enum>(level));

        logger_ = std::make_shared<spdlog::logger>("nxs", console_sink);
        logger_->set_level(static_cast<spdlog::level::level_enum>(level));

        // Set as default logger
        spdlog::set_default_logger(logger_);

        // Set pattern: [timestamp] [level] message
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
#else
        level_ = level;
#endif
    }

    // Initialize logger with file output
    void init_file(const std::string& filename,
                   Level level = Level::Info,
                   size_t max_size = 1024 * 1024 * 10,  // 10MB
                   size_t max_files = 3) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            filename, max_size, max_files
        );
        file_sink->set_level(static_cast<spdlog::level::level_enum>(level));

        logger_ = std::make_shared<spdlog::logger>("nxs", file_sink);
        logger_->set_level(static_cast<spdlog::level::level_enum>(level));

        spdlog::set_default_logger(logger_);
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
#else
        (void)filename;
        (void)max_size;
        (void)max_files;
        level_ = level;
        std::cout << "[WARN] File logging not available without spdlog - using console only\n";
#endif
    }

    // Initialize logger with both console and file output
    void init_combined(const std::string& filename,
                       Level console_level = Level::Info,
                       Level file_level = Level::Debug) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(static_cast<spdlog::level::level_enum>(console_level));

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename);
        file_sink->set_level(static_cast<spdlog::level::level_enum>(file_level));

        spdlog::sinks_init_list sink_list = {console_sink, file_sink};
        logger_ = std::make_shared<spdlog::logger>("nxs", sink_list);
        logger_->set_level(spdlog::level::trace);  // Capture all, sinks will filter

        spdlog::set_default_logger(logger_);
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
#else
        (void)filename;
        (void)file_level;
        level_ = console_level;
        std::cout << "[WARN] File logging not available without spdlog - using console only\n";
#endif
    }

    // Set log level
    void set_level(Level level) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) {
            logger_->set_level(static_cast<spdlog::level::level_enum>(level));
        }
#else
        level_ = level;
#endif
    }

    // Log functions
    template<typename... Args>
    void trace(const char* fmt, Args&&... args) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) logger_->trace(fmt, std::forward<Args>(args)...);
#else
        log_fallback(Level::Trace, fmt, std::forward<Args>(args)...);
#endif
    }

    template<typename... Args>
    void debug(const char* fmt, Args&&... args) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) logger_->debug(fmt, std::forward<Args>(args)...);
#else
        log_fallback(Level::Debug, fmt, std::forward<Args>(args)...);
#endif
    }

    template<typename... Args>
    void info(const char* fmt, Args&&... args) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) logger_->info(fmt, std::forward<Args>(args)...);
#else
        log_fallback(Level::Info, fmt, std::forward<Args>(args)...);
#endif
    }

    template<typename... Args>
    void warn(const char* fmt, Args&&... args) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) logger_->warn(fmt, std::forward<Args>(args)...);
#else
        log_fallback(Level::Warn, fmt, std::forward<Args>(args)...);
#endif
    }

    template<typename... Args>
    void error(const char* fmt, Args&&... args) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) logger_->error(fmt, std::forward<Args>(args)...);
#else
        log_fallback(Level::Error, fmt, std::forward<Args>(args)...);
#endif
    }

    template<typename... Args>
    void critical(const char* fmt, Args&&... args) {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) logger_->critical(fmt, std::forward<Args>(args)...);
#else
        log_fallback(Level::Critical, fmt, std::forward<Args>(args)...);
#endif
    }

    // Flush logs
    void flush() {
#ifdef NEXUSSIM_HAVE_SPDLOG
        if (logger_) logger_->flush();
#else
        std::cout.flush();
#endif
    }

#ifdef NEXUSSIM_HAVE_SPDLOG
    // Get underlying spdlog logger
    std::shared_ptr<spdlog::logger> get_logger() {
        return logger_;
    }
#endif

private:
    Logger() {
        // Default initialization with console
        init_console();
    }

    ~Logger() {
        flush();
    }

    // Delete copy/move constructors
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

#ifndef NEXUSSIM_HAVE_SPDLOG
    // Fallback logging without spdlog
    static const char* level_to_string(Level lvl) {
        switch (lvl) {
            case Level::Trace: return "TRACE";
            case Level::Debug: return "DEBUG";
            case Level::Info: return "INFO";
            case Level::Warn: return "WARN";
            case Level::Error: return "ERROR";
            case Level::Critical: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }

    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ) % 1000;

        std::tm tm_buf;
        #ifdef _WIN32
        localtime_s(&tm_buf, &time);
        #else
        localtime_r(&time, &tm_buf);
        #endif

        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    template<typename... Args>
    void log_fallback(Level lvl, const char* fmt, Args&&... args) {
        if (static_cast<int>(lvl) < static_cast<int>(level_)) {
            return;
        }

        std::ostringstream oss;
        oss << "[" << get_timestamp() << "] "
            << "[" << level_to_string(lvl) << "] ";

        // Simple format string handling - just print the format string and args
        // This is a simplified version without actual format string parsing
        if constexpr (sizeof...(args) == 0) {
            oss << fmt;
        } else {
            // For simplicity, we'll use a basic implementation
            // that doesn't fully support spdlog's format syntax
            oss << fmt;
            ((oss << " " << args), ...);
        }

        std::cout << oss.str() << std::endl;
    }

    Level level_ = Level::Info;
#endif

#ifdef NEXUSSIM_HAVE_SPDLOG
    std::shared_ptr<spdlog::logger> logger_;
#endif
};

// ============================================================================
// Convenience Macros
// ============================================================================

#define NXS_LOG_TRACE(...)    ::nxs::Logger::instance().trace(__VA_ARGS__)
#define NXS_LOG_DEBUG(...)    ::nxs::Logger::instance().debug(__VA_ARGS__)
#define NXS_LOG_INFO(...)     ::nxs::Logger::instance().info(__VA_ARGS__)
#define NXS_LOG_WARN(...)     ::nxs::Logger::instance().warn(__VA_ARGS__)
#define NXS_LOG_ERROR(...)    ::nxs::Logger::instance().error(__VA_ARGS__)
#define NXS_LOG_CRITICAL(...) ::nxs::Logger::instance().critical(__VA_ARGS__)

// ============================================================================
// Scoped Timer for Performance Logging
// ============================================================================

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name)
        , start_(std::chrono::high_resolution_clock::now())
    {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start_
        ).count();

        NXS_LOG_DEBUG("{} took {} ms", name_, duration);
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#define NXS_SCOPED_TIMER(name) ::nxs::ScopedTimer _timer_##__LINE__(name)
#define NXS_FUNCTION_TIMER() NXS_SCOPED_TIMER(__func__)

} // namespace nxs
