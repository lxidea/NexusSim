#pragma once

#include <nexussim/core/types.hpp>
#include <nexussim/core/exception.hpp>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace nxs {

// ============================================================================
// Memory Alignment Utilities
// ============================================================================

namespace memory {

// Alignment constants
inline constexpr std::size_t DefaultAlignment = 64;  // Cache line size
inline constexpr std::size_t GPUAlignment = 256;     // GPU memory alignment

// Align a size to the next multiple of alignment
inline constexpr std::size_t align_size(std::size_t size, std::size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Check if a pointer is aligned
template<typename T>
inline bool is_aligned(T* ptr, std::size_t alignment = DefaultAlignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// Allocate aligned memory
inline void* allocate_aligned(std::size_t size, std::size_t alignment = DefaultAlignment) {
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        throw std::bad_alloc();
    }
#endif
    return ptr;
}

// Free aligned memory
inline void free_aligned(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace memory

// ============================================================================
// Memory Arena - Fast Stack-like Allocator
// ============================================================================

class MemoryArena {
public:
    explicit MemoryArena(std::size_t block_size = 1024 * 1024)  // 1MB default
        : block_size_(block_size)
        , current_block_(nullptr)
        , current_offset_(0)
    {
        allocate_new_block();
    }

    ~MemoryArena() {
        for (auto* block : blocks_) {
            memory::free_aligned(block);
        }
    }

    // Delete copy/move operations
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) = delete;
    MemoryArena& operator=(MemoryArena&&) = delete;

    // Allocate memory from the arena
    void* allocate(std::size_t size, std::size_t alignment = memory::DefaultAlignment) {
        // Align the current offset
        std::size_t aligned_offset = memory::align_size(current_offset_, alignment);

        // Check if we need a new block
        if (aligned_offset + size > block_size_) {
            allocate_new_block();
            aligned_offset = 0;
        }

        void* ptr = static_cast<char*>(current_block_) + aligned_offset;
        current_offset_ = aligned_offset + size;

        return ptr;
    }

    // Allocate and construct object
    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        void* ptr = allocate(sizeof(T), alignof(T));
        return new (ptr) T(std::forward<Args>(args)...);
    }

    // Reset the arena (doesn't free memory, just resets offset)
    void reset() {
        current_offset_ = 0;
        if (!blocks_.empty()) {
            current_block_ = blocks_[0];
        }
    }

    // Get total allocated size
    std::size_t total_allocated() const {
        return blocks_.size() * block_size_;
    }

    // Get used size
    std::size_t used() const {
        return (blocks_.size() - 1) * block_size_ + current_offset_;
    }

private:
    void allocate_new_block() {
        void* new_block = memory::allocate_aligned(block_size_, memory::DefaultAlignment);
        blocks_.push_back(new_block);
        current_block_ = new_block;
        current_offset_ = 0;
    }

    std::size_t block_size_;
    std::vector<void*> blocks_;
    void* current_block_;
    std::size_t current_offset_;
};

// ============================================================================
// Memory Pool - Fixed-Size Object Allocator
// ============================================================================

template<typename T, std::size_t ChunkSize = 1024>
class MemoryPool {
public:
    MemoryPool() : free_list_(nullptr) {
        allocate_chunk();
    }

    ~MemoryPool() {
        for (auto* chunk : chunks_) {
            memory::free_aligned(chunk);
        }
    }

    // Delete copy/move operations
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    // Allocate one object
    T* allocate() {
        if (!free_list_) {
            allocate_chunk();
        }

        Node* node = free_list_;
        free_list_ = node->next;
        return reinterpret_cast<T*>(node);
    }

    // Deallocate one object
    void deallocate(T* ptr) {
        Node* node = reinterpret_cast<Node*>(ptr);
        node->next = free_list_;
        free_list_ = node;
    }

    // Construct object
    template<typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        return new (ptr) T(std::forward<Args>(args)...);
    }

    // Destroy object
    void destroy(T* ptr) {
        ptr->~T();
        deallocate(ptr);
    }

    // Get statistics
    std::size_t total_allocated() const {
        return chunks_.size() * ChunkSize;
    }

private:
    union Node {
        T object;
        Node* next;
    };

    void allocate_chunk() {
        // Allocate aligned chunk
        void* chunk = memory::allocate_aligned(
            ChunkSize * sizeof(Node),
            std::max(alignof(T), memory::DefaultAlignment)
        );

        chunks_.push_back(chunk);

        // Build free list
        Node* nodes = static_cast<Node*>(chunk);
        for (std::size_t i = 0; i < ChunkSize - 1; ++i) {
            nodes[i].next = &nodes[i + 1];
        }
        nodes[ChunkSize - 1].next = free_list_;
        free_list_ = nodes;
    }

    std::vector<void*> chunks_;
    Node* free_list_;
};

// ============================================================================
// Smart Buffer - RAII Wrapper for Aligned Memory
// ============================================================================

template<typename T>
class AlignedBuffer {
public:
    AlignedBuffer() : data_(nullptr), size_(0), capacity_(0) {}

    explicit AlignedBuffer(std::size_t size, std::size_t alignment = memory::DefaultAlignment)
        : data_(nullptr), size_(size), capacity_(size)
    {
        if (size > 0) {
            data_ = static_cast<T*>(memory::allocate_aligned(size * sizeof(T), alignment));
        }
    }

    ~AlignedBuffer() {
        if (data_) {
            // Destroy objects if T is non-trivial
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (std::size_t i = 0; i < size_; ++i) {
                    data_[i].~T();
                }
            }
            memory::free_aligned(data_);
        }
    }

    // Delete copy, allow move
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                memory::free_aligned(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // Accessors
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return size_; }
    std::size_t capacity() const { return capacity_; }

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    T& at(std::size_t i) {
        NXS_CHECK_RANGE(i, size_);
        return data_[i];
    }

    const T& at(std::size_t i) const {
        NXS_CHECK_RANGE(i, size_);
        return data_[i];
    }

    // Iterators
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }

    // Resize (simple version, doesn't preserve data)
    void resize(std::size_t new_size, std::size_t alignment = memory::DefaultAlignment) {
        if (new_size > capacity_) {
            if (data_) {
                memory::free_aligned(data_);
            }
            data_ = static_cast<T*>(memory::allocate_aligned(new_size * sizeof(T), alignment));
            capacity_ = new_size;
        }
        size_ = new_size;
    }

    // Fill with value
    void fill(const T& value) {
        std::fill(begin(), end(), value);
    }

    // Zero memory
    void zero() {
        std::memset(data_, 0, size_ * sizeof(T));
    }

private:
    T* data_;
    std::size_t size_;
    std::size_t capacity_;
};

// ============================================================================
// Memory Statistics
// ============================================================================

struct MemoryStats {
    std::size_t total_allocated = 0;
    std::size_t total_freed = 0;
    std::size_t current_usage = 0;
    std::size_t peak_usage = 0;
    std::size_t allocation_count = 0;
};

// Global memory tracker (optional, for debugging)
class MemoryTracker {
public:
    static MemoryTracker& instance() {
        static MemoryTracker tracker;
        return tracker;
    }

    void record_allocation(std::size_t size) {
        stats_.total_allocated += size;
        stats_.current_usage += size;
        stats_.allocation_count++;
        if (stats_.current_usage > stats_.peak_usage) {
            stats_.peak_usage = stats_.current_usage;
        }
    }

    void record_deallocation(std::size_t size) {
        stats_.total_freed += size;
        stats_.current_usage -= size;
    }

    const MemoryStats& stats() const { return stats_; }
    void reset() { stats_ = MemoryStats{}; }

private:
    MemoryTracker() = default;
    MemoryStats stats_;
};

} // namespace nxs
