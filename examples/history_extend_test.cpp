/**
 * @file history_extend_test.cpp
 * @brief Wave 23: Verify history vector extension from 32 to 64 slots
 *
 * Tests:
 * 1. NEXUSSIM_HISTORY_SIZE constant equals 64
 * 2. MaterialState history array has 64 elements
 * 3. Slots 0-31 work as before
 * 4. Slots 32-63 are accessible and zero-initialized
 * 5. Checkpoint round-trip preserves all 64 slots
 * 6. sizeof(MaterialState) reflects extended history
 * 7. sizeof(MaterialStateData) matches extended history
 * 8. All slots survive write→read cycle
 */

#include <nexussim/physics/material.hpp>
#include <nexussim/io/checkpoint.hpp>
#include <cmath>
#include <iostream>
#include <cstring>

static int test_count = 0;
static int pass_count = 0;

#define CHECK(expr) do { \
    test_count++; \
    if (!(expr)) { \
        std::cerr << "FAIL [" << test_count << "]: " << #expr \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
    } else { pass_count++; } \
} while(0)

#define CHECK_NEAR(a, b, tol) do { \
    test_count++; \
    double _a = static_cast<double>(a), _b = static_cast<double>(b); \
    if (std::fabs(_a - _b) > (tol)) { \
        std::cerr << "FAIL [" << test_count << "]: " << #a << " ≈ " << #b \
                  << " (got " << _a << " vs " << _b << ", tol=" << (tol) \
                  << ") (" << __FILE__ << ":" << __LINE__ << ")\n"; \
    } else { pass_count++; } \
} while(0)

int main() {
    using namespace nxs;
    using namespace nxs::physics;

    std::cout << "=== Wave 23: History Extension Test ===" << std::endl;

    // ---- Test 1: Constant value ----
    std::cout << "\n[Test 1] NEXUSSIM_HISTORY_SIZE = 64\n";
    CHECK(MaterialState::NEXUSSIM_HISTORY_SIZE == 64);

    // ---- Test 2: Array size ----
    std::cout << "[Test 2] history array has 64 elements\n";
    CHECK(sizeof(MaterialState{}.history) == 64 * sizeof(Real));

    // ---- Test 3: Slots 0-31 work as before ----
    std::cout << "[Test 3] Slots 0-31 read/write\n";
    {
        MaterialState ms;
        for (int i = 0; i < 32; ++i) {
            ms.history[i] = static_cast<Real>(i * 1.1);
        }
        for (int i = 0; i < 32; ++i) {
            CHECK_NEAR(ms.history[i], i * 1.1, 1.0e-10);
        }
    }

    // ---- Test 4: Slots 32-63 accessible and zero-initialized ----
    std::cout << "[Test 4] Slots 32-63 zero-initialized and writable\n";
    {
        MaterialState ms;
        // Check zero-initialized
        for (int i = 32; i < 64; ++i) {
            CHECK_NEAR(ms.history[i], 0.0, 1.0e-15);
        }
        // Write and read back
        for (int i = 32; i < 64; ++i) {
            ms.history[i] = static_cast<Real>(100.0 + i);
        }
        for (int i = 32; i < 64; ++i) {
            CHECK_NEAR(ms.history[i], 100.0 + i, 1.0e-10);
        }
    }

    // ---- Test 5: Checkpoint round-trip ----
    std::cout << "[Test 5] MaterialStateData round-trip preserves all 64 slots\n";
    {
        MaterialState original;
        for (int i = 0; i < 64; ++i) {
            original.history[i] = static_cast<Real>(i * 3.14159);
        }
        original.plastic_strain = 0.05;
        original.temperature = 500.0;
        original.damage = 0.3;

        // Serialize
        io::MaterialStateData data;
        data.from_material_state(original);

        // Deserialize
        MaterialState restored;
        data.to_material_state(restored);

        for (int i = 0; i < 64; ++i) {
            CHECK_NEAR(restored.history[i], original.history[i], 1.0e-12);
        }
        CHECK_NEAR(restored.plastic_strain, 0.05, 1.0e-12);
        CHECK_NEAR(restored.temperature, 500.0, 1.0e-12);
        CHECK_NEAR(restored.damage, 0.3, 1.0e-12);
    }

    // ---- Test 6: sizeof(MaterialState) reflects extended history ----
    std::cout << "[Test 6] sizeof(MaterialState) includes 64 history slots\n";
    {
        // history contributes 64 * sizeof(Real) bytes
        size_t history_bytes = 64 * sizeof(Real);
        CHECK(sizeof(MaterialState) >= history_bytes);
    }

    // ---- Test 7: sizeof(MaterialStateData) matches ----
    std::cout << "[Test 7] MaterialStateData history array matches\n";
    {
        io::MaterialStateData msd;
        CHECK(sizeof(msd.history) == 64 * sizeof(Real));
    }

    // ---- Test 8: Full pattern - fill all 64, serialize, verify ----
    std::cout << "[Test 8] Full 64-slot pattern survives serialize cycle\n";
    {
        MaterialState ms;
        // Use distinctive pattern: slot i = sin(i) * 1000
        for (int i = 0; i < 64; ++i) {
            ms.history[i] = static_cast<Real>(std::sin(static_cast<double>(i)) * 1000.0);
        }

        io::MaterialStateData data;
        data.from_material_state(ms);

        MaterialState ms2;
        data.to_material_state(ms2);

        bool all_match = true;
        for (int i = 0; i < 64; ++i) {
            if (std::fabs(static_cast<double>(ms2.history[i] - ms.history[i])) > 1.0e-10) {
                all_match = false;
                std::cerr << "  Slot " << i << " mismatch: " << ms2.history[i]
                          << " vs " << ms.history[i] << "\n";
            }
        }
        CHECK(all_match);
    }

    // ---- Summary ----
    std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===\n";
    return (pass_count == test_count) ? 0 : 1;
}
