/**
 * @file eos_wave43_test.cpp
 * @brief Wave 43: EOS models test suite (7 new EOS models, ~56 tests)
 *
 * Tests 7 EOS models (~8 tests each):
 *  1. LSZK          - Lee-Szekely-Kung reactive burn with Arrhenius kinetics
 *  2. NASG           - Noble-Abel-Stiffened-Gas (multi-phase)
 *  3. Puff           - Porous material solid/vapor/mixed-phase
 *  4. Exponential    - JWL-like A*exp(-R1*V) + B*exp(-R2*V) + C/(omega*V)
 *  5. IdealGasVT     - Volume-temperature coupled ideal gas
 *  6. Compaction2    - 2nd-gen compaction with separate K_unload
 *  7. CompactionTab  - Fully tabulated compaction curves
 */

#include <nexussim/physics/eos_wave43.hpp>
#include <iostream>
#include <cmath>

using namespace nxs;
using namespace nxs::physics;

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg << "\n"; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    double _va = static_cast<double>(a); \
    double _vb = static_cast<double>(b); \
    double _vt = static_cast<double>(tol); \
    if (std::abs(_va - _vb) <= _vt) { tests_passed++; } \
    else { tests_failed++; std::cout << "[FAIL] " << msg \
        << " (got " << _va << ", expected " << _vb << ", tol " << _vt << ")\n"; } \
} while(0)

// ============================================================================
// 1. LSZK EOS Tests
// ============================================================================

static void test_lszk_eos() {
    EOSWave43Properties props;
    props.type    = EOSWave43Type::LSZK;
    props.rho0    = 1700.0;   // RDX-like density [kg/m^3]
    props.lszk_K_u     = 1.0e10;  // Unreacted bulk modulus
    props.lszk_gamma_u = 3.0;
    props.lszk_gamma_p = 2.7;
    props.lszk_e0_p    = 3.68e6;  // Reference energy for products [J/kg]
    props.lszk_A_arr   = 1.0e6;   // Moderate burn rate
    props.lszk_Ea_R    = 5000.0;  // E_a/R = 5000 K

    // (a) Fully unreacted (lambda=0): pressure from unreacted Tait EOS only
    {
        EOSWave43State state;
        state.lambda = 0.0;
        // eta = 1.1 => mu = 0.1 => P = K_u * 0.1 = 1e9
        Real rho = 1.1 * props.rho0;
        Real P = LSZKEOS::compute_pressure(props, rho, 5.0e6, state);
        CHECK_NEAR(P, 1.0e9, 1.0e6, "LSZK lambda=0 unreacted Tait pressure");
    }

    // (b) Fully burned (lambda=1): products pressure dominates
    {
        EOSWave43State state;
        state.lambda = 1.0;
        Real rho = 1700.0;
        Real e   = 6.0e6;  // above e0_p
        Real e_eff = e - props.lszk_e0_p;  // 2.32e6
        Real P_expected = (props.lszk_gamma_p - 1.0) * rho * e_eff;  // 1.7*2.32e6*1700
        Real P = LSZKEOS::compute_pressure(props, rho, e, state);
        CHECK_NEAR(P, P_expected, P_expected * 1.0e-10, "LSZK lambda=1 products pressure");
    }

    // (c) Mixed state (lambda=0.5): mixture pressure
    {
        EOSWave43State state;
        state.lambda = 0.5;
        Real rho = 1700.0;
        Real e   = 6.0e6;
        // P_u at reference density = K_u*(1-1) = 0
        // P_p = (2.7-1)*1700*(6e6-3.68e6) = 1.7*1700*2.32e6
        Real P_p = (props.lszk_gamma_p - 1.0) * rho * (e - props.lszk_e0_p);
        Real P_expected = 0.5 * P_p;  // P_u=0 at rho=rho0
        Real P = LSZKEOS::compute_pressure(props, rho, e, state);
        CHECK_NEAR(P, P_expected, P_expected * 1.0e-10, "LSZK lambda=0.5 mixed pressure");
    }

    // (d) Stateless overload defaults to lambda=1
    {
        Real rho = 1700.0;
        Real e   = 6.0e6;
        EOSWave43State s1; s1.lambda = 1.0;
        Real P_state    = LSZKEOS::compute_pressure(props, rho, e, s1);
        Real P_stateless = LSZKEOS::compute_pressure(props, rho, e);
        CHECK_NEAR(P_stateless, P_state, 1.0, "LSZK stateless == fully-burned state");
    }

    // (e) Zero density returns zero pressure
    {
        EOSWave43State state;
        state.lambda = 0.5;
        Real P = LSZKEOS::compute_pressure(props, 0.0, 1.0e6, state);
        CHECK_NEAR(P, 0.0, 1.0, "LSZK zero density => zero pressure");
    }

    // (f) Burn rate update: lambda should increase
    {
        EOSWave43State state;
        state.lambda = 0.0;
        Real e  = 5.0e6;   // T = 5e6/1000 = 5000 K (using nominal Cv=1000)
        Real dt = 1.0e-7;
        LSZKEOS::update_burn(props, e, dt, state);
        CHECK(state.lambda > 0.0, "LSZK burn update: lambda increases");
        CHECK(state.lambda <= 1.0, "LSZK burn update: lambda stays <= 1");
    }

    // (g) Burn capped at 1.0
    {
        EOSWave43State state;
        state.lambda = 0.99;
        // Very large dt with high T should push lambda over 1, but it should clamp
        LSZKEOS::update_burn(props, 1.0e9, 1.0, state);
        CHECK(state.lambda <= 1.0, "LSZK burn capped at 1.0");
    }

    // (h) Sound speed positive for compressed state
    {
        EOSWave43State state; state.lambda = 1.0;
        Real c = LSZKEOS::sound_speed(props, 1700.0, 6.0e6, state);
        CHECK(c > 0.0, "LSZK sound speed positive");
    }
}

// ============================================================================
// 2. NASG EOS Tests
// ============================================================================

static void test_nasg_eos() {
    EOSWave43Properties props;
    props.type       = EOSWave43Type::NASG;
    props.rho0       = 1.2;      // Air-like density
    props.nasg_gamma = 1.4;
    props.nasg_b     = 0.0;
    props.nasg_q     = 0.0;
    props.nasg_p_inf = 0.0;

    // (a) Ideal gas limit (b=0, p_inf=0, q=0): P = (gamma-1)*rho*e
    {
        Real rho = 1.2;
        Real e   = 200000.0;  // ~ 200 kJ/kg
        Real P_expected = (1.4 - 1.0) * 1.2 * 200000.0;  // 96000 Pa
        Real P = NASGEOS::compute_pressure(props, rho, e);
        CHECK_NEAR(P, P_expected, 1.0, "NASG ideal gas limit pressure");
    }

    // (b) Standard air at reference: P ~ 101325 Pa
    {
        Real rho = 1.2;
        Real Cv  = 718.0;
        Real T   = 293.15;
        Real e   = Cv * T;  // specific internal energy
        Real P_expected = (1.4 - 1.0) * rho * e;  // = rho * R * T = rho * (Cp-Cv) * T
        Real P = NASGEOS::compute_pressure(props, rho, e);
        CHECK_NEAR(P, P_expected, 1.0, "NASG standard air pressure");
    }

    // (c) Covolume correction: with b>0, pressure should be higher than ideal
    {
        props.nasg_b = 1.0e-3;   // 1 L/kg covolume
        Real rho = 100.0;        // Dense gas
        Real e   = 200000.0;
        Real P_no_b  = (1.4 - 1.0) * rho * e;
        Real P_with_b = NASGEOS::compute_pressure(props, rho, e);
        CHECK(P_with_b > P_no_b, "NASG covolume increases pressure");
        props.nasg_b = 0.0;      // Reset
    }

    // (d) p_inf correction: with p_inf>0, equilibrium pressure is shifted
    {
        props.nasg_p_inf = 1.0e8;  // 100 MPa stiffness (water-like)
        props.nasg_gamma = 7.15;
        Real rho = 1000.0;
        Real e   = 1.0e5;
        Real P = NASGEOS::compute_pressure(props, rho, e);
        Real P_stiff_term = -props.nasg_gamma * props.nasg_p_inf;
        CHECK(P < P_stiff_term + 2.0 * std::abs(P_stiff_term), "NASG p_inf shifts pressure");
        props.nasg_p_inf = 0.0;
        props.nasg_gamma = 1.4;
    }

    // (e) q offset shifts energy reference
    {
        props.nasg_q = 50000.0;  // 50 kJ/kg reference energy
        Real rho = 1.2;
        Real e1  = 200000.0;
        Real e2  = e1 - props.nasg_q;
        Real P1 = NASGEOS::compute_pressure(props, rho, e1);
        Real P2_no_q = (1.4 - 1.0) * rho * e2;
        CHECK_NEAR(P1, P2_no_q, 1.0, "NASG q offset equivalent to energy shift");
        props.nasg_q = 0.0;
    }

    // (f) Zero energy gives zero pressure (no other contributions)
    {
        Real P = NASGEOS::compute_pressure(props, 1.2, 0.0);
        CHECK_NEAR(P, 0.0, 1.0, "NASG zero energy => zero pressure");
    }

    // (g) Sound speed positive
    {
        Real c = NASGEOS::sound_speed(props, 1.2, 200000.0);
        CHECK(c > 0.0, "NASG sound speed positive");
        // For ideal gas: c = sqrt(gamma * R * T) ~ 340 m/s for air
        CHECK(c > 200.0 && c < 600.0, "NASG sound speed in physical range for air");
    }

    // (h) Pressure scales linearly with density (ideal gas limit)
    {
        Real e = 200000.0;
        Real P1 = NASGEOS::compute_pressure(props, 1.2, e);
        Real P2 = NASGEOS::compute_pressure(props, 2.4, e);
        CHECK_NEAR(P2 / P1, 2.0, 1.0e-10, "NASG pressure scales linearly with density");
    }
}

// ============================================================================
// 3. Puff EOS Tests
// ============================================================================

static void test_puff_eos() {
    EOSWave43Properties props;
    props.type           = EOSWave43Type::Puff;
    props.rho0           = 2000.0;   // Porous aluminum-like (porosity alpha0 ~0.74)
    props.puff_rho_solid = 2700.0;   // Dense aluminum
    props.puff_K_s       = 7.5e10;   // Solid bulk modulus
    props.puff_E_sub     = 1.0e7;    // Sublimation energy [J/kg]
    props.puff_alpha0    = 2000.0 / 2700.0;
    props.puff_gamma_v   = 5.0 / 3.0;  // Monatomic vapor

    // (a) Solid compression: rho > rho_solid and e < E_sub -> P from Tait
    {
        Real rho = 2700.0;  // Fully compacted
        Real e   = 1.0e5;   // Low energy (below E_sub)
        Real eta_s = rho / props.puff_rho_solid;  // = 1.0
        Real P_expected = props.puff_K_s * (eta_s - 1.0);  // = 0 at reference
        Real P = PuffEOS::compute_pressure(props, rho, e);
        CHECK_NEAR(P, P_expected, 1.0e4, "Puff solid at rho_solid: P ~ 0");
    }

    // (b) Solid compression above rho_solid
    {
        Real rho = 2.9 * 1000.0;  // 2900 kg/m^3 (above 2700)
        Real e   = 1.0e5;
        Real eta_s = rho / props.puff_rho_solid;
        Real P_expected = props.puff_K_s * (eta_s - 1.0);
        Real P = PuffEOS::compute_pressure(props, rho, e);
        CHECK(P > 0.0, "Puff solid compression: P > 0 above rho_solid");
        CHECK_NEAR(P, P_expected, P_expected * 0.01, "Puff solid compression Tait P");
    }

    // (c) Vapor phase: rho < rho0, e > E_sub -> P from gamma-law
    {
        Real rho = 500.0;   // Low density (expanded)
        Real e   = 2.0e7;   // Above sublimation energy
        Real e_vap = e - props.puff_E_sub;
        Real P_expected = (props.puff_gamma_v - 1.0) * rho * e_vap;
        Real P = PuffEOS::compute_pressure(props, rho, e);
        CHECK_NEAR(P, P_expected, P_expected * 0.01, "Puff vapor phase pressure");
    }

    // (d) Zero density returns zero
    {
        Real P = PuffEOS::compute_pressure(props, 0.0, 1.0e5);
        CHECK_NEAR(P, 0.0, 1.0, "Puff zero density => zero pressure");
    }

    // (e) Solid phase: higher density gives higher pressure
    {
        Real P1 = PuffEOS::compute_pressure(props, 2800.0, 1.0e5);
        Real P2 = PuffEOS::compute_pressure(props, 2900.0, 1.0e5);
        CHECK(P2 > P1, "Puff solid: higher density => higher pressure");
    }

    // (f) Vapor phase: higher energy gives higher pressure
    {
        Real P1 = PuffEOS::compute_pressure(props, 500.0, 1.5e7);
        Real P2 = PuffEOS::compute_pressure(props, 500.0, 2.0e7);
        CHECK(P2 > P1, "Puff vapor: higher energy => higher pressure");
    }

    // (g) Mixed regime: energy between 0 and E_sub, rho between rho0 and rho_solid
    {
        Real rho = 2350.0;  // Between rho0=2000 and rho_solid=2700
        Real e   = 5.0e6;   // Half of E_sub
        Real P = PuffEOS::compute_pressure(props, rho, e);
        CHECK(P >= 0.0, "Puff mixed phase pressure non-negative");
    }

    // (h) Sound speed positive for solid regime
    {
        Real c = PuffEOS::sound_speed(props, 2800.0, 1.0e5);
        CHECK(c > 0.0, "Puff solid sound speed positive");
    }
}

// ============================================================================
// 4. Exponential (JWL-like) EOS Tests
// ============================================================================

static void test_exponential_eos() {
    EOSWave43Properties props;
    props.type      = EOSWave43Type::Exponential;
    props.rho0      = 1630.0;    // PETN-like initial density
    // Standard PETN JWL parameters
    props.exp_A     = 6.170e11;  // [Pa]
    props.exp_B     = 1.691e10;  // [Pa]
    props.exp_C     = 1.339e9;   // [Pa]
    props.exp_R1    = 4.40;
    props.exp_R2    = 1.20;
    props.exp_omega = 0.25;
    props.exp_V0    = 1.0 / 1630.0;

    // (a) At reference density (r=V/V0=1): compute expected value
    {
        Real rho = 1630.0;
        Real e   = 0.0;  // No energy contribution
        Real r   = 1.0;  // V/V0 = rho0/rho = 1
        Real P_expected = props.exp_A * std::exp(-props.exp_R1 * r)
                        + props.exp_B * std::exp(-props.exp_R2 * r)
                        + props.exp_C / (props.exp_omega * r)
                        + props.exp_omega * rho * e;
        Real P = ExponentialEOS::compute_pressure(props, rho, e);
        CHECK_NEAR(P, P_expected, P_expected * 1.0e-10, "Exponential EOS at reference density");
    }

    // (b) Compressed state (rho > rho0): higher pressure
    {
        Real P_ref   = ExponentialEOS::compute_pressure(props, props.rho0, 0.0);
        Real P_comp  = ExponentialEOS::compute_pressure(props, props.rho0 * 1.5, 0.0);
        // At higher density: r = V/V0 < 1, exp terms are larger
        CHECK(P_comp != P_ref, "Exponential EOS: compressed state differs from reference");
    }

    // (c) Energy term adds to pressure
    {
        Real rho = 1630.0;
        Real P_no_e = ExponentialEOS::compute_pressure(props, rho, 0.0);
        Real P_with_e = ExponentialEOS::compute_pressure(props, rho, 1.0e6);
        Real delta = props.exp_omega * rho * 1.0e6;
        CHECK_NEAR(P_with_e - P_no_e, delta, 1.0, "Exponential EOS energy contribution");
    }

    // (d) stateless no-energy overload matches compute_pressure with e=0
    {
        Real rho = 1630.0;
        Real P_full = ExponentialEOS::compute_pressure(props, rho, 0.0);
        Real P_ne   = ExponentialEOS::compute_pressure_no_energy(props, rho);
        CHECK_NEAR(P_full, P_ne, 1.0, "Exponential EOS no-energy overload matches e=0");
    }

    // (e) Zero density returns zero
    {
        Real P = ExponentialEOS::compute_pressure(props, 0.0, 0.0);
        CHECK_NEAR(P, 0.0, 1.0, "Exponential EOS zero density => zero");
    }

    // (f) Unified dispatch: Exponential type
    {
        Real P_direct   = ExponentialEOS::compute_pressure(props, 1630.0, 1.0e6);
        Real P_dispatch = EOSWave43::compute_pressure(props, 1630.0, 1.0e6);
        CHECK_NEAR(P_direct, P_dispatch, 1.0, "Exponential EOS unified dispatch");
    }

    // (g) Sound speed positive
    {
        Real c = ExponentialEOS::sound_speed(props, 1630.0, 1.0e6);
        CHECK(c > 0.0, "Exponential EOS sound speed positive");
    }

    // (h) Expansion (r>1, rho<rho0): exponential terms become small
    {
        Real P_ref  = ExponentialEOS::compute_pressure_no_energy(props, props.rho0);
        Real P_exp  = ExponentialEOS::compute_pressure_no_energy(props, props.rho0 * 0.1);
        // At r=10, exp(-R1*10) ~ exp(-44) ~ 0; C term dominates but is small
        CHECK(P_exp < P_ref, "Exponential EOS expanded state < reference");
    }
}

// ============================================================================
// 5. IdealGasVT EOS Tests
// ============================================================================

static void test_idealgas_vt_eos() {
    EOSWave43Properties props;
    props.type      = EOSWave43Type::IdealGasVT;
    props.rho0      = 1.2;    // Air density [kg/m^3]
    props.igt_R_gas = 287.0;  // Specific gas constant for air [J/(kg*K)]
    props.igt_Cv    = 718.0;  // Cv for air [J/(kg*K)]
    props.igt_T0    = 293.15; // Reference temperature [K]

    // (a) Standard atmosphere: P = rho * R * T = 1.2 * 287 * 293.15 ~ 100966 Pa
    {
        Real rho = 1.2;
        Real T   = 293.15;
        Real P_expected = rho * props.igt_R_gas * T;
        Real P = IdealGasVTEOS::compute_pressure_T(props, rho, T);
        CHECK_NEAR(P, P_expected, 1.0, "IdealGasVT standard atmosphere pressure");
        // Should be close to 1 atm
        CHECK(P > 90000.0 && P < 120000.0, "IdealGasVT: ~1 atm at standard conditions");
    }

    // (b) compute_pressure using energy: e = Cv * T => T = e/Cv
    {
        Real rho = 1.2;
        Real T   = 293.15;
        Real e   = props.igt_Cv * T;
        Real P_T = IdealGasVTEOS::compute_pressure_T(props, rho, T);
        Real P_e = IdealGasVTEOS::compute_pressure(props, rho, e);
        CHECK_NEAR(P_T, P_e, 1.0, "IdealGasVT T-based and e-based pressure agree");
    }

    // (c) Temperature update from energy
    {
        EOSWave43State state;
        state.temperature = 0.0;
        Real e_new = 718.0 * 500.0;  // T = 500 K
        IdealGasVTEOS::update_temperature(props, e_new, state);
        CHECK_NEAR(state.temperature, 500.0, 1.0e-10, "IdealGasVT temperature update");
    }

    // (d) Zero energy gives zero pressure
    {
        Real P = IdealGasVTEOS::compute_pressure(props, 1.2, 0.0);
        CHECK_NEAR(P, 0.0, 1.0, "IdealGasVT zero energy => zero pressure");
    }

    // (e) Pressure scales linearly with density
    {
        Real e = 718.0 * 293.15;
        Real P1 = IdealGasVTEOS::compute_pressure(props, 1.2, e);
        Real P2 = IdealGasVTEOS::compute_pressure(props, 2.4, e);
        // Note: e = Cv*T is specific (per kg), so rho doubles, P doubles
        CHECK_NEAR(P2 / P1, 2.0, 1.0e-10, "IdealGasVT linear density scaling");
    }

    // (f) Pressure scales linearly with energy (temperature)
    {
        Real rho = 1.2;
        Real e1  = 718.0 * 293.15;
        Real e2  = 718.0 * 586.30;  // Double temperature
        Real P1 = IdealGasVTEOS::compute_pressure(props, rho, e1);
        Real P2 = IdealGasVTEOS::compute_pressure(props, rho, e2);
        CHECK_NEAR(P2 / P1, 2.0, 1.0e-10, "IdealGasVT linear energy scaling");
    }

    // (g) Sound speed: c = sqrt(gamma * R * T)
    {
        Real e = 718.0 * 293.15;
        Real c = IdealGasVTEOS::sound_speed(props, 1.2, e);
        Real gamma = (718.0 + 287.0) / 718.0;  // ~ 1.4
        Real c_expected = std::sqrt(gamma * 287.0 * 293.15);  // ~ 340 m/s
        CHECK_NEAR(c, c_expected, 1.0, "IdealGasVT sound speed value");
        CHECK(c > 300.0 && c < 400.0, "IdealGasVT sound speed in physical range");
    }

    // (h) Unified dispatch works for IdealGasVT
    {
        Real e   = 718.0 * 293.15;
        Real P1  = IdealGasVTEOS::compute_pressure(props, 1.2, e);
        Real P2  = EOSWave43::compute_pressure(props, 1.2, e);
        CHECK_NEAR(P1, P2, 1.0, "IdealGasVT unified dispatch");
    }
}

// ============================================================================
// 6. Compaction2 EOS Tests
// ============================================================================

static void test_compaction2_eos() {
    EOSWave43Properties props;
    props.type               = EOSWave43Type::Compaction2;
    props.rho0               = 1000.0;  // Porous powder initial density
    props.compact2_K_unload  = 5.0e9;   // Unloading modulus

    // Build simple loading curve: P(mu) = 1e9 * mu  (linear, modulus = 1 GPa)
    props.compact2_curve.add_point(0.0, 0.0);
    props.compact2_curve.add_point(0.1, 1.0e8);
    props.compact2_curve.add_point(0.2, 2.0e8);
    props.compact2_curve.add_point(0.5, 5.0e8);
    props.compact2_curve.add_point(1.0, 1.0e9);

    // (a) Reference state (rho=rho0, mu=0): zero pressure
    {
        Real P = Compaction2EOS::compute_pressure(props, 1000.0, 0.0);
        CHECK_NEAR(P, 0.0, 1.0, "Compaction2 at reference density: P=0");
    }

    // (b) Loading path: mu=0.1 => P = 1e8 from curve
    {
        EOSWave43State state;
        state.mu_max = 0.0;
        Real rho = 1100.0;  // mu = 0.1
        Real P = Compaction2EOS::compute_pressure(props, rho, 0.0, state);
        CHECK_NEAR(P, 1.0e8, 1.0, "Compaction2 loading path P(mu=0.1)");
    }

    // (c) State updated on loading: mu_max tracks maximum
    {
        EOSWave43State state;
        state.mu_max = 0.0;
        Real rho = 1200.0;  // mu = 0.2
        Compaction2EOS::compute_pressure(props, rho, 0.0, state);
        CHECK_NEAR(state.mu_max, 0.2, 1.0e-12, "Compaction2 state.mu_max updated on loading");
    }

    // (d) Unloading path: pressure from (mu_max, P_max) with K_unload slope
    {
        EOSWave43State state;
        state.mu_max = 0.2;
        state.p_max  = 2.0e8;  // Pre-set from prior loading
        // Now unload to mu = 0.1
        Real rho = 1100.0;  // mu = 0.1
        Real P_expected = 2.0e8 - 5.0e9 * (0.2 - 0.1);  // = 2e8 - 5e8 = -3e8 => clamped to 0
        Real P = Compaction2EOS::compute_pressure(props, rho, 0.0, state);
        // With K_unload=5e9 and delta_mu=0.1, drop is 5e8 which exceeds p_max=2e8 => 0
        CHECK_NEAR(P, 0.0, 1.0, "Compaction2 unloading below zero clamped");
    }

    // (e) Tension (mu<0): zero pressure
    {
        Real P = Compaction2EOS::compute_pressure(props, 800.0, 0.0);
        CHECK_NEAR(P, 0.0, 1.0, "Compaction2 tension: P=0");
    }

    // (f) Stateless overload always on loading path
    {
        Real P1 = Compaction2EOS::compute_pressure(props, 1100.0, 0.0);
        EOSWave43State s; s.mu_max = 0.0;
        Real P2 = Compaction2EOS::compute_pressure(props, 1100.0, 0.0, s);
        CHECK_NEAR(P1, P2, 1.0, "Compaction2 stateless == loading path");
    }

    // (g) Sound speed on loading path
    {
        Real c = Compaction2EOS::sound_speed(props, 1100.0, 0.0);
        CHECK(c > 0.0, "Compaction2 sound speed positive on loading");
    }

    // (h) Unified dispatch
    {
        Real P1 = Compaction2EOS::compute_pressure(props, 1100.0, 0.0);
        Real P2 = EOSWave43::compute_pressure(props, 1100.0, 0.0);
        CHECK_NEAR(P1, P2, 1.0, "Compaction2 unified dispatch");
    }
}

// ============================================================================
// 7. CompactionTab EOS Tests
// ============================================================================

static void test_compaction_tab_eos() {
    EOSWave43Properties props;
    props.type              = EOSWave43Type::CompactionTab;
    props.rho0              = 1000.0;
    props.comptab_mu_max_ref = 0.5;

    // Build loading curve: P_load(mu) = 2e9 * mu (linear)
    props.comptab_load.set(0, 0.0, 0.0);
    props.comptab_load.set(1, 0.1, 2.0e8);
    props.comptab_load.set(2, 0.2, 4.0e8);
    props.comptab_load.set(3, 0.5, 1.0e9);
    props.comptab_load.n = 4;

    // Build unloading curve at mu_max_ref=0.5: linear from (0, 0) to (0.5, 1e9)
    // but with slope = 1e9/0.5 * 0.8 (80% stiffness => softer unloading)
    props.comptab_unload.set(0, 0.0, 0.0);
    props.comptab_unload.set(1, 0.1, 1.6e8);
    props.comptab_unload.set(2, 0.2, 3.2e8);
    props.comptab_unload.set(3, 0.5, 8.0e8);
    props.comptab_unload.n = 4;

    // (a) Loading path: P at mu=0.1 matches load curve
    {
        EOSWave43State state;
        state.mu_max = 0.0;
        Real P = CompactionTabEOS::compute_pressure(props, 1100.0, 0.0, state);
        CHECK_NEAR(P, 2.0e8, 1.0, "CompactionTab loading P(mu=0.1)");
    }

    // (b) State updated on loading
    {
        EOSWave43State state;
        state.mu_max = 0.0;
        CompactionTabEOS::compute_pressure(props, 1200.0, 0.0, state);  // mu=0.2
        CHECK_NEAR(state.mu_max, 0.2, 1.0e-12, "CompactionTab mu_max updated on loading");
    }

    // (c) Reference density: zero pressure
    {
        Real P = CompactionTabEOS::compute_pressure(props, 1000.0, 0.0);
        CHECK_NEAR(P, 0.0, 1.0, "CompactionTab at rho0: P=0");
    }

    // (d) Tension: zero pressure
    {
        Real P = CompactionTabEOS::compute_pressure(props, 900.0, 0.0);
        CHECK_NEAR(P, 0.0, 1.0, "CompactionTab tension: P=0");
    }

    // (e) Unloading gives non-negative pressure
    {
        EOSWave43State state;
        state.mu_max = 0.3;
        state.p_max  = props.comptab_load.evaluate(0.3);
        // Unload to mu=0.1
        Real P = CompactionTabEOS::compute_pressure(props, 1100.0, 0.0, state);
        CHECK(P >= 0.0, "CompactionTab unload pressure non-negative");
    }

    // (f) Stateless overload (loading path only)
    {
        Real P1 = CompactionTabEOS::compute_pressure(props, 1200.0, 0.0);
        CHECK_NEAR(P1, 4.0e8, 1.0, "CompactionTab stateless loading path P(mu=0.2)");
    }

    // (g) Sound speed positive
    {
        Real c = CompactionTabEOS::sound_speed(props, 1200.0, 0.0);
        CHECK(c > 0.0, "CompactionTab sound speed positive on loading");
    }

    // (h) Unified dispatch
    {
        Real P1 = CompactionTabEOS::compute_pressure(props, 1100.0, 0.0);
        Real P2 = EOSWave43::compute_pressure(props, 1100.0, 0.0);
        CHECK_NEAR(P1, P2, 1.0, "CompactionTab unified dispatch");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Wave 43 EOS Test Suite ===\n\n";

    std::cout << "--- LSZK (Reactive Burn) EOS ---\n";
    test_lszk_eos();

    std::cout << "--- NASG (Noble-Abel-Stiffened-Gas) EOS ---\n";
    test_nasg_eos();

    std::cout << "--- Puff (Porous Material) EOS ---\n";
    test_puff_eos();

    std::cout << "--- Exponential (JWL-like) EOS ---\n";
    test_exponential_eos();

    std::cout << "--- IdealGasVT (Volume-Temperature) EOS ---\n";
    test_idealgas_vt_eos();

    std::cout << "--- Compaction2 (2nd-Gen Compaction) EOS ---\n";
    test_compaction2_eos();

    std::cout << "--- CompactionTab (Tabulated Compaction) EOS ---\n";
    test_compaction_tab_eos();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return (tests_failed > 0) ? 1 : 0;
}
