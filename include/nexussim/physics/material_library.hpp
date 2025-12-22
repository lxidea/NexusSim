#pragma once

/**
 * @file material_library.hpp
 * @brief Comprehensive library of common engineering materials
 *
 * This library provides pre-defined material properties for commonly used
 * engineering materials. All properties are at room temperature (20°C)
 * unless otherwise specified.
 */

#include <nexussim/physics/material.hpp>
#include <string>
#include <unordered_map>
#include <memory>

namespace nxs {
namespace physics {

/**
 * @brief Enumeration of predefined materials
 */
enum class MaterialName {
    // Structural Steels
    Steel_MildSteel,              ///< Mild steel (AISI 1020)
    Steel_StructuralSteel,        ///< Structural steel (A36)
    Steel_StainlessSteel304,      ///< Stainless steel 304
    Steel_StainlessSteel316,      ///< Stainless steel 316
    Steel_HighStrengthSteel,      ///< High strength steel (A572 Grade 50)
    Steel_ToolSteel,              ///< Tool steel (D2)

    // Aluminum Alloys
    Aluminum_1100,                ///< Aluminum 1100 (pure aluminum)
    Aluminum_2024_T3,             ///< Aluminum 2024-T3 (aerospace)
    Aluminum_6061_T6,             ///< Aluminum 6061-T6 (general purpose)
    Aluminum_7075_T6,             ///< Aluminum 7075-T6 (high strength)

    // Titanium Alloys
    Titanium_Grade2,              ///< Commercially pure titanium
    Titanium_Ti6Al4V,             ///< Ti-6Al-4V (aerospace grade)

    // Copper Alloys
    Copper_Pure,                  ///< Pure copper
    Copper_Brass,                 ///< Brass (70Cu-30Zn)
    Copper_Bronze,                ///< Bronze (90Cu-10Sn)

    // Polymers
    Polymer_ABS,                  ///< ABS plastic
    Polymer_Nylon6,               ///< Nylon 6 (PA6)
    Polymer_Polycarbonate,        ///< Polycarbonate (PC)
    Polymer_PEEK,                 ///< PEEK (high performance)
    Polymer_Acrylic,              ///< Acrylic (PMMA)

    // Composites
    Composite_CarbonFiberEpoxy,   ///< Carbon fiber/epoxy (unidirectional)
    Composite_GlassFiberEpoxy,    ///< Glass fiber/epoxy (unidirectional)

    // Concrete and Ceramics
    Concrete_Normal,              ///< Normal strength concrete
    Concrete_HighStrength,        ///< High strength concrete
    Ceramic_Alumina,              ///< Alumina (Al2O3)
    Ceramic_SiliconCarbide,       ///< Silicon carbide (SiC)

    // Other Materials
    Rubber_Natural,               ///< Natural rubber
    Wood_Oak,                     ///< Oak wood (along grain)
    Glass_SodaLime,               ///< Soda-lime glass

    // Generic/Test Materials
    Generic_Soft,                 ///< Soft material for testing
    Generic_Medium,               ///< Medium stiffness for testing
    Generic_Rigid                 ///< Rigid material for testing
};

/**
 * @brief Material Library - Database of engineering materials
 */
class MaterialLibrary {
public:
    /**
     * @brief Get material properties by name
     */
    static MaterialProperties get(MaterialName name) {
        MaterialProperties props;

        switch (name) {
            // ================================================================
            // STEELS
            // ================================================================

            case MaterialName::Steel_MildSteel:
                props.density = 7850.0;              // kg/m³
                props.E = 200.0e9;                   // Pa (200 GPa)
                props.nu = 0.29;
                props.yield_stress = 250.0e6;        // Pa (250 MPa)
                props.specific_heat = 486.0;         // J/(kg·K)
                props.thermal_expansion = 11.7e-6;   // 1/K
                props.thermal_conductivity = 51.9;   // W/(m·K)
                break;

            case MaterialName::Steel_StructuralSteel:
                props.density = 7850.0;
                props.E = 200.0e9;
                props.nu = 0.30;
                props.yield_stress = 250.0e6;
                props.specific_heat = 486.0;
                props.thermal_expansion = 12.0e-6;
                props.thermal_conductivity = 52.0;
                break;

            case MaterialName::Steel_StainlessSteel304:
                props.density = 8000.0;
                props.E = 193.0e9;
                props.nu = 0.29;
                props.yield_stress = 215.0e6;
                props.specific_heat = 500.0;
                props.thermal_expansion = 17.3e-6;
                props.thermal_conductivity = 16.2;
                break;

            case MaterialName::Steel_StainlessSteel316:
                props.density = 8000.0;
                props.E = 193.0e9;
                props.nu = 0.30;
                props.yield_stress = 290.0e6;
                props.specific_heat = 500.0;
                props.thermal_expansion = 16.0e-6;
                props.thermal_conductivity = 16.3;
                break;

            case MaterialName::Steel_HighStrengthSteel:
                props.density = 7850.0;
                props.E = 200.0e9;
                props.nu = 0.30;
                props.yield_stress = 345.0e6;        // A572 Grade 50
                props.specific_heat = 486.0;
                props.thermal_expansion = 11.7e-6;
                props.thermal_conductivity = 51.9;
                break;

            case MaterialName::Steel_ToolSteel:
                props.density = 7800.0;
                props.E = 210.0e9;
                props.nu = 0.30;
                props.yield_stress = 2000.0e6;       // D2 hardened
                props.specific_heat = 460.0;
                props.thermal_expansion = 11.5e-6;
                props.thermal_conductivity = 20.0;
                break;

            // ================================================================
            // ALUMINUM ALLOYS
            // ================================================================

            case MaterialName::Aluminum_1100:
                props.density = 2710.0;
                props.E = 69.0e9;
                props.nu = 0.33;
                props.yield_stress = 34.0e6;
                props.specific_heat = 904.0;
                props.thermal_expansion = 23.6e-6;
                props.thermal_conductivity = 222.0;
                break;

            case MaterialName::Aluminum_2024_T3:
                props.density = 2780.0;
                props.E = 73.0e9;
                props.nu = 0.33;
                props.yield_stress = 345.0e6;
                props.specific_heat = 875.0;
                props.thermal_expansion = 22.9e-6;
                props.thermal_conductivity = 121.0;
                break;

            case MaterialName::Aluminum_6061_T6:
                props.density = 2700.0;
                props.E = 69.0e9;
                props.nu = 0.33;
                props.yield_stress = 276.0e6;
                props.specific_heat = 896.0;
                props.thermal_expansion = 23.6e-6;
                props.thermal_conductivity = 167.0;
                break;

            case MaterialName::Aluminum_7075_T6:
                props.density = 2810.0;
                props.E = 71.7e9;
                props.nu = 0.33;
                props.yield_stress = 503.0e6;
                props.specific_heat = 860.0;
                props.thermal_expansion = 23.2e-6;
                props.thermal_conductivity = 130.0;
                break;

            // ================================================================
            // TITANIUM ALLOYS
            // ================================================================

            case MaterialName::Titanium_Grade2:
                props.density = 4510.0;
                props.E = 105.0e9;
                props.nu = 0.34;
                props.yield_stress = 275.0e6;
                props.specific_heat = 523.0;
                props.thermal_expansion = 8.6e-6;
                props.thermal_conductivity = 16.4;
                break;

            case MaterialName::Titanium_Ti6Al4V:
                props.density = 4430.0;
                props.E = 114.0e9;
                props.nu = 0.34;
                props.yield_stress = 880.0e6;
                props.specific_heat = 526.0;
                props.thermal_expansion = 8.6e-6;
                props.thermal_conductivity = 6.7;
                break;

            // ================================================================
            // COPPER ALLOYS
            // ================================================================

            case MaterialName::Copper_Pure:
                props.density = 8960.0;
                props.E = 130.0e9;
                props.nu = 0.34;
                props.yield_stress = 70.0e6;
                props.specific_heat = 385.0;
                props.thermal_expansion = 16.5e-6;
                props.thermal_conductivity = 401.0;
                break;

            case MaterialName::Copper_Brass:
                props.density = 8530.0;
                props.E = 100.0e9;
                props.nu = 0.34;
                props.yield_stress = 125.0e6;
                props.specific_heat = 380.0;
                props.thermal_expansion = 20.5e-6;
                props.thermal_conductivity = 120.0;
                break;

            case MaterialName::Copper_Bronze:
                props.density = 8800.0;
                props.E = 110.0e9;
                props.nu = 0.34;
                props.yield_stress = 170.0e6;
                props.specific_heat = 377.0;
                props.thermal_expansion = 18.0e-6;
                props.thermal_conductivity = 50.0;
                break;

            // ================================================================
            // POLYMERS
            // ================================================================

            case MaterialName::Polymer_ABS:
                props.density = 1050.0;
                props.E = 2.3e9;
                props.nu = 0.35;
                props.yield_stress = 40.0e6;
                props.specific_heat = 1386.0;
                props.thermal_expansion = 73.8e-6;
                props.thermal_conductivity = 0.25;
                break;

            case MaterialName::Polymer_Nylon6:
                props.density = 1140.0;
                props.E = 2.9e9;
                props.nu = 0.39;
                props.yield_stress = 50.0e6;
                props.specific_heat = 1700.0;
                props.thermal_expansion = 80.0e-6;
                props.thermal_conductivity = 0.25;
                break;

            case MaterialName::Polymer_Polycarbonate:
                props.density = 1200.0;
                props.E = 2.4e9;
                props.nu = 0.37;
                props.yield_stress = 62.0e6;
                props.specific_heat = 1250.0;
                props.thermal_expansion = 65.0e-6;
                props.thermal_conductivity = 0.20;
                break;

            case MaterialName::Polymer_PEEK:
                props.density = 1320.0;
                props.E = 3.6e9;
                props.nu = 0.40;
                props.yield_stress = 90.0e6;
                props.specific_heat = 1340.0;
                props.thermal_expansion = 47.0e-6;
                props.thermal_conductivity = 0.25;
                break;

            case MaterialName::Polymer_Acrylic:
                props.density = 1190.0;
                props.E = 3.2e9;
                props.nu = 0.35;
                props.yield_stress = 72.0e6;
                props.specific_heat = 1420.0;
                props.thermal_expansion = 70.0e-6;
                props.thermal_conductivity = 0.19;
                break;

            // ================================================================
            // COMPOSITES (Simplified isotropic approximations)
            // ================================================================

            case MaterialName::Composite_CarbonFiberEpoxy:
                props.density = 1600.0;
                props.E = 70.0e9;                    // Approximate (anisotropic in reality)
                props.nu = 0.30;
                props.yield_stress = 600.0e6;
                props.specific_heat = 1050.0;
                props.thermal_expansion = 1.0e-6;    // Very low
                props.thermal_conductivity = 5.0;
                break;

            case MaterialName::Composite_GlassFiberEpoxy:
                props.density = 1850.0;
                props.E = 38.0e9;                    // Approximate
                props.nu = 0.28;
                props.yield_stress = 440.0e6;
                props.specific_heat = 1200.0;
                props.thermal_expansion = 8.0e-6;
                props.thermal_conductivity = 0.3;
                break;

            // ================================================================
            // CONCRETE AND CERAMICS
            // ================================================================

            case MaterialName::Concrete_Normal:
                props.density = 2400.0;
                props.E = 30.0e9;
                props.nu = 0.20;
                props.yield_stress = 4.0e6;          // Compressive strength ~30 MPa
                props.specific_heat = 880.0;
                props.thermal_expansion = 10.0e-6;
                props.thermal_conductivity = 1.4;
                break;

            case MaterialName::Concrete_HighStrength:
                props.density = 2500.0;
                props.E = 40.0e9;
                props.nu = 0.20;
                props.yield_stress = 8.0e6;          // Compressive strength ~60 MPa
                props.specific_heat = 880.0;
                props.thermal_expansion = 10.0e-6;
                props.thermal_conductivity = 1.6;
                break;

            case MaterialName::Ceramic_Alumina:
                props.density = 3950.0;
                props.E = 380.0e9;
                props.nu = 0.22;
                props.yield_stress = 2000.0e6;       // Brittle - compressive strength
                props.specific_heat = 880.0;
                props.thermal_expansion = 8.1e-6;
                props.thermal_conductivity = 30.0;
                break;

            case MaterialName::Ceramic_SiliconCarbide:
                props.density = 3210.0;
                props.E = 410.0e9;
                props.nu = 0.14;
                props.yield_stress = 3500.0e6;
                props.specific_heat = 750.0;
                props.thermal_expansion = 4.0e-6;
                props.thermal_conductivity = 120.0;
                break;

            // ================================================================
            // OTHER MATERIALS
            // ================================================================

            case MaterialName::Rubber_Natural:
                props.density = 920.0;
                props.E = 0.01e9;                    // Very soft (10 MPa)
                props.nu = 0.48;                     // Nearly incompressible
                props.yield_stress = 15.0e6;
                props.specific_heat = 1900.0;
                props.thermal_expansion = 200.0e-6;
                props.thermal_conductivity = 0.13;
                break;

            case MaterialName::Wood_Oak:
                props.density = 750.0;
                props.E = 12.0e9;                    // Along grain
                props.nu = 0.30;
                props.yield_stress = 50.0e6;
                props.specific_heat = 2400.0;
                props.thermal_expansion = 54.0e-6;
                props.thermal_conductivity = 0.17;
                break;

            case MaterialName::Glass_SodaLime:
                props.density = 2500.0;
                props.E = 69.0e9;
                props.nu = 0.23;
                props.yield_stress = 50.0e6;         // Brittle
                props.specific_heat = 840.0;
                props.thermal_expansion = 9.0e-6;
                props.thermal_conductivity = 1.05;
                break;

            // ================================================================
            // GENERIC/TEST MATERIALS
            // ================================================================

            case MaterialName::Generic_Soft:
                props.density = 1000.0;
                props.E = 1.0e6;                     // 1 MPa
                props.nu = 0.30;
                props.yield_stress = 0.1e6;
                props.specific_heat = 1000.0;
                props.thermal_expansion = 10.0e-6;
                props.thermal_conductivity = 1.0;
                break;

            case MaterialName::Generic_Medium:
                props.density = 2000.0;
                props.E = 10.0e9;                    // 10 GPa
                props.nu = 0.30;
                props.yield_stress = 100.0e6;
                props.specific_heat = 1000.0;
                props.thermal_expansion = 10.0e-6;
                props.thermal_conductivity = 10.0;
                break;

            case MaterialName::Generic_Rigid:
                props.density = 8000.0;
                props.E = 200.0e9;                   // 200 GPa
                props.nu = 0.30;
                props.yield_stress = 1000.0e6;
                props.specific_heat = 500.0;
                props.thermal_expansion = 10.0e-6;
                props.thermal_conductivity = 50.0;
                break;

            default:
                // Return default material
                break;
        }

        // Compute derived properties (G, K, sound_speed)
        props.compute_derived();

        return props;
    }

    /**
     * @brief Get material name as string
     */
    static std::string to_string(MaterialName name) {
        switch (name) {
            case MaterialName::Steel_MildSteel: return "Mild Steel (AISI 1020)";
            case MaterialName::Steel_StructuralSteel: return "Structural Steel (A36)";
            case MaterialName::Steel_StainlessSteel304: return "Stainless Steel 304";
            case MaterialName::Steel_StainlessSteel316: return "Stainless Steel 316";
            case MaterialName::Steel_HighStrengthSteel: return "High Strength Steel (A572-50)";
            case MaterialName::Steel_ToolSteel: return "Tool Steel (D2)";

            case MaterialName::Aluminum_1100: return "Aluminum 1100";
            case MaterialName::Aluminum_2024_T3: return "Aluminum 2024-T3";
            case MaterialName::Aluminum_6061_T6: return "Aluminum 6061-T6";
            case MaterialName::Aluminum_7075_T6: return "Aluminum 7075-T6";

            case MaterialName::Titanium_Grade2: return "Titanium Grade 2";
            case MaterialName::Titanium_Ti6Al4V: return "Titanium Ti-6Al-4V";

            case MaterialName::Copper_Pure: return "Pure Copper";
            case MaterialName::Copper_Brass: return "Brass (70Cu-30Zn)";
            case MaterialName::Copper_Bronze: return "Bronze (90Cu-10Sn)";

            case MaterialName::Polymer_ABS: return "ABS Plastic";
            case MaterialName::Polymer_Nylon6: return "Nylon 6 (PA6)";
            case MaterialName::Polymer_Polycarbonate: return "Polycarbonate";
            case MaterialName::Polymer_PEEK: return "PEEK";
            case MaterialName::Polymer_Acrylic: return "Acrylic (PMMA)";

            case MaterialName::Composite_CarbonFiberEpoxy: return "Carbon Fiber/Epoxy";
            case MaterialName::Composite_GlassFiberEpoxy: return "Glass Fiber/Epoxy";

            case MaterialName::Concrete_Normal: return "Normal Concrete";
            case MaterialName::Concrete_HighStrength: return "High Strength Concrete";
            case MaterialName::Ceramic_Alumina: return "Alumina (Al2O3)";
            case MaterialName::Ceramic_SiliconCarbide: return "Silicon Carbide (SiC)";

            case MaterialName::Rubber_Natural: return "Natural Rubber";
            case MaterialName::Wood_Oak: return "Oak Wood";
            case MaterialName::Glass_SodaLime: return "Soda-Lime Glass";

            case MaterialName::Generic_Soft: return "Generic Soft Material";
            case MaterialName::Generic_Medium: return "Generic Medium Material";
            case MaterialName::Generic_Rigid: return "Generic Rigid Material";

            default: return "Unknown Material";
        }
    }

    /**
     * @brief Print material properties summary
     */
    static void print_properties(MaterialName name) {
        auto props = get(name);

        std::cout << "Material: " << to_string(name) << "\n";
        std::cout << "  Density: " << props.density << " kg/m³\n";
        std::cout << "  Young's Modulus: " << props.E/1e9 << " GPa\n";
        std::cout << "  Poisson's Ratio: " << props.nu << "\n";
        std::cout << "  Shear Modulus: " << props.G/1e9 << " GPa\n";
        std::cout << "  Bulk Modulus: " << props.K/1e9 << " GPa\n";
        std::cout << "  Yield Stress: " << props.yield_stress/1e6 << " MPa\n";
        std::cout << "  Sound Speed: " << props.sound_speed << " m/s\n";
        std::cout << "  Thermal Expansion: " << props.thermal_expansion*1e6 << " µm/(m·K)\n";
        std::cout << "  Thermal Conductivity: " << props.thermal_conductivity << " W/(m·K)\n";
    }
};

} // namespace physics
} // namespace nxs
