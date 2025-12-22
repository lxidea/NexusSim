/**
 * @file material_library_demo.cpp
 * @brief Demonstration of the Material Library functionality
 */

#include <nexussim/nexussim.hpp>
#include <nexussim/physics/material_library.hpp>
#include <iostream>
#include <iomanip>

using namespace nxs;
using namespace nxs::physics;

int main() {
    std::cout << "=================================================\n";
    std::cout << "NexusSim Material Library Demonstration\n";
    std::cout << "=================================================\n\n";

    // Demonstrate different material categories

    std::cout << "========== STRUCTURAL STEELS ==========\n\n";
    MaterialLibrary::print_properties(MaterialName::Steel_MildSteel);
    std::cout << "\n";
    MaterialLibrary::print_properties(MaterialName::Steel_StainlessSteel316);
    std::cout << "\n";

    std::cout << "========== ALUMINUM ALLOYS ==========\n\n";
    MaterialLibrary::print_properties(MaterialName::Aluminum_6061_T6);
    std::cout << "\n";
    MaterialLibrary::print_properties(MaterialName::Aluminum_7075_T6);
    std::cout << "\n";

    std::cout << "========== TITANIUM ALLOYS ==========\n\n";
    MaterialLibrary::print_properties(MaterialName::Titanium_Ti6Al4V);
    std::cout << "\n";

    std::cout << "========== POLYMERS ==========\n\n";
    MaterialLibrary::print_properties(MaterialName::Polymer_ABS);
    std::cout << "\n";
    MaterialLibrary::print_properties(MaterialName::Polymer_PEEK);
    std::cout << "\n";

    std::cout << "========== COMPOSITES ==========\n\n";
    MaterialLibrary::print_properties(MaterialName::Composite_CarbonFiberEpoxy);
    std::cout << "\n";

    std::cout << "========== MATERIAL COMPARISON TABLE ==========\n\n";

    // Create comparison table
    std::cout << std::setw(30) << std::left << "Material"
              << std::setw(12) << "Density"
              << std::setw(12) << "E (GPa)"
              << std::setw(10) << "ν"
              << std::setw(15) << "σ_y (MPa)"
              << std::setw(15) << "c (m/s)"
              << "\n";
    std::cout << std::string(95, '-') << "\n";

    MaterialName materials[] = {
        MaterialName::Steel_MildSteel,
        MaterialName::Aluminum_6061_T6,
        MaterialName::Titanium_Ti6Al4V,
        MaterialName::Copper_Pure,
        MaterialName::Polymer_ABS,
        MaterialName::Polymer_PEEK,
        MaterialName::Composite_CarbonFiberEpoxy,
        MaterialName::Concrete_Normal,
        MaterialName::Wood_Oak,
        MaterialName::Rubber_Natural
    };

    for (auto mat : materials) {
        auto props = MaterialLibrary::get(mat);
        std::cout << std::setw(30) << std::left << MaterialLibrary::to_string(mat)
                  << std::setw(12) << std::fixed << std::setprecision(0) << props.density
                  << std::setw(12) << std::setprecision(1) << props.E/1e9
                  << std::setw(10) << std::setprecision(3) << props.nu
                  << std::setw(15) << std::setprecision(0) << props.yield_stress/1e6
                  << std::setw(15) << std::setprecision(0) << props.sound_speed
                  << "\n";
    }

    std::cout << "\n========== USAGE EXAMPLE ==========\n\n";

    // Example: Compare wave speeds for crash simulation
    std::cout << "Wave Speed Comparison (for explicit time step calculation):\n\n";

    MaterialName crash_materials[] = {
        MaterialName::Steel_MildSteel,
        MaterialName::Aluminum_6061_T6,
        MaterialName::Polymer_ABS
    };

    for (auto mat : crash_materials) {
        auto props = MaterialLibrary::get(mat);
        Real dt_crit = 0.01 / props.sound_speed;  // Characteristic length = 1cm

        std::cout << "  " << std::setw(25) << std::left << MaterialLibrary::to_string(mat)
                  << ": c = " << std::setw(8) << std::fixed << std::setprecision(1) << props.sound_speed
                  << " m/s,  Δt_crit = " << std::scientific << std::setprecision(2) << dt_crit << " s\n";
    }

    std::cout << "\n========== STRENGTH-TO-WEIGHT COMPARISON ==========\n\n";

    std::cout << "Specific Strength (Yield Stress / Density) Ranking:\n\n";

    struct MaterialRanking {
        MaterialName name;
        Real specific_strength;
    };

    std::vector<MaterialRanking> rankings;
    for (auto mat : materials) {
        auto props = MaterialLibrary::get(mat);
        rankings.push_back({mat, props.yield_stress / props.density});
    }

    // Sort by specific strength
    std::sort(rankings.begin(), rankings.end(),
              [](const MaterialRanking& a, const MaterialRanking& b) {
                  return a.specific_strength > b.specific_strength;
              });

    int rank = 1;
    for (const auto& r : rankings) {
        std::cout << "  " << rank++ << ". " << std::setw(30) << std::left
                  << MaterialLibrary::to_string(r.name)
                  << ": " << std::fixed << std::setprecision(0)
                  << r.specific_strength/1e3 << " kPa/(kg/m³)\n";
    }

    std::cout << "\n========== STIFFNESS-TO-WEIGHT COMPARISON ==========\n\n";

    std::cout << "Specific Stiffness (E / Density) Ranking:\n\n";

    std::vector<MaterialRanking> stiffness_rankings;
    for (auto mat : materials) {
        auto props = MaterialLibrary::get(mat);
        stiffness_rankings.push_back({mat, props.E / props.density});
    }

    std::sort(stiffness_rankings.begin(), stiffness_rankings.end(),
              [](const MaterialRanking& a, const MaterialRanking& b) {
                  return a.specific_strength > b.specific_strength;
              });

    rank = 1;
    for (const auto& r : stiffness_rankings) {
        std::cout << "  " << rank++ << ". " << std::setw(30) << std::left
                  << MaterialLibrary::to_string(r.name)
                  << ": " << std::scientific << std::setprecision(2)
                  << r.specific_strength << " Pa/(kg/m³)\n";
    }

    std::cout << "\n=================================================\n";
    std::cout << "Material Library Contains " << sizeof(materials)/sizeof(materials[0])
              << "+ Engineering Materials\n";
    std::cout << "=================================================\n";

    return 0;
}
