/**
 * @file large_deformation_test.cpp
 * @brief Test suite for large deformation mechanics module
 *
 * Tests:
 * 1. Deformation gradient computation
 * 2. Green-Lagrange strain
 * 3. Euler-Almansi strain
 * 4. Stress transformations (Cauchy <-> 2nd PK)
 * 5. Rate of deformation tensor
 * 6. Internal force computation (Total vs Updated Lagrangian)
 * 7. Geometry update
 */

#include <nexussim/physics/large_deformation.hpp>
#include <nexussim/physics/material.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>

using namespace nxs;
using namespace nxs::physics;

// Test tolerance
constexpr Real TOL = 1.0e-10;

int tests_passed = 0;
int tests_total = 0;

void check(bool condition, const std::string& test_name) {
    tests_total++;
    if (condition) {
        tests_passed++;
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name << "\n";
    }
}

// Test 1: Tensor utilities
void test_tensor_utilities() {
    std::cout << "\n=== Test 1: Tensor Utilities ===\n";

    // Test determinant
    Real A[9] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
    Real det = tensor::determinant3x3(A);
    check(std::abs(det - 1.0) < TOL, "Determinant computation");

    // Test inverse
    Real A_inv[9];
    tensor::inverse3x3(A, A_inv);

    // A * A_inv should be identity
    Real I[9];
    tensor::multiply3x3(A, A_inv, I);
    bool is_identity = std::abs(I[0] - 1) < TOL && std::abs(I[4] - 1) < TOL &&
                       std::abs(I[8] - 1) < TOL && std::abs(I[1]) < TOL &&
                       std::abs(I[2]) < TOL && std::abs(I[3]) < TOL;
    check(is_identity, "Matrix inverse (A * A^-1 = I)");

    // Test transpose
    Real A_T[9];
    tensor::transpose3x3(A, A_T);
    check(A_T[1] == A[3] && A_T[3] == A[1], "Matrix transpose");

    // Test Voigt conversion
    Real sym[9] = {1, 2, 3, 2, 4, 5, 3, 5, 6};
    Real voigt[6];
    tensor::symmetric_to_voigt(sym, voigt);
    check(voigt[0] == 1 && voigt[1] == 4 && voigt[2] == 6 &&
          voigt[3] == 2 && voigt[4] == 5 && voigt[5] == 3,
          "Symmetric to Voigt conversion");

    Real sym2[9];
    tensor::voigt_to_symmetric(voigt, sym2);
    check(sym2[0] == sym[0] && sym2[4] == sym[4] && sym2[8] == sym[8],
          "Voigt to symmetric conversion");
}

// Test 2: Deformation gradient (identity case)
void test_deformation_gradient_identity() {
    std::cout << "\n=== Test 2: Deformation Gradient (Identity) ===\n";

    // 8-node hex element with zero displacement
    constexpr int num_nodes = 8;
    Real dN_dX[num_nodes * 3] = {0};  // Will set to some values
    Real disp[num_nodes * 3] = {0};   // Zero displacement

    // Simple shape function derivatives for unit cube
    // At center (xi=0, eta=0, zeta=0)
    const Real a = 0.125;
    dN_dX[0*3+0] = -a; dN_dX[0*3+1] = -a; dN_dX[0*3+2] = -a;
    dN_dX[1*3+0] =  a; dN_dX[1*3+1] = -a; dN_dX[1*3+2] = -a;
    dN_dX[2*3+0] =  a; dN_dX[2*3+1] =  a; dN_dX[2*3+2] = -a;
    dN_dX[3*3+0] = -a; dN_dX[3*3+1] =  a; dN_dX[3*3+2] = -a;
    dN_dX[4*3+0] = -a; dN_dX[4*3+1] = -a; dN_dX[4*3+2] =  a;
    dN_dX[5*3+0] =  a; dN_dX[5*3+1] = -a; dN_dX[5*3+2] =  a;
    dN_dX[6*3+0] =  a; dN_dX[6*3+1] =  a; dN_dX[6*3+2] =  a;
    dN_dX[7*3+0] = -a; dN_dX[7*3+1] =  a; dN_dX[7*3+2] =  a;

    Real F[9];
    compute_deformation_gradient(dN_dX, disp, num_nodes, F);

    // Should be identity matrix
    bool is_identity = std::abs(F[0] - 1) < TOL && std::abs(F[4] - 1) < TOL &&
                       std::abs(F[8] - 1) < TOL && std::abs(F[1]) < TOL &&
                       std::abs(F[2]) < TOL && std::abs(F[3]) < TOL;
    check(is_identity, "F = I for zero displacement");

    Real J = tensor::determinant3x3(F);
    check(std::abs(J - 1.0) < TOL, "J = 1 for zero displacement");
}

// Test 3: Deformation gradient (uniform stretch)
void test_deformation_gradient_stretch() {
    std::cout << "\n=== Test 3: Deformation Gradient (Uniform Stretch) ===\n";

    // Unit cube nodes [-1,1]^3 (natural coordinates scale)
    constexpr int num_nodes = 8;
    Real coords[num_nodes * 3] = {
        -1, -1, -1,  1, -1, -1,  1, 1, -1,  -1, 1, -1,  // bottom face
        -1, -1,  1,  1, -1,  1,  1, 1,  1,  -1, 1,  1   // top face
    };

    // 10% stretch in x direction: u_x = 0.1 * X
    Real stretch = 0.1;
    Real disp[num_nodes * 3] = {0};
    for (int n = 0; n < num_nodes; ++n) {
        disp[n*3 + 0] = stretch * coords[n*3 + 0];  // u_x = 0.1 * X
    }

    // Shape function derivatives at element center for unit cube [-1,1]^3
    // For element [-1,1]^3, dN/dX = dN/dξ (Jacobian = I)
    const Real a = 0.125;
    Real dN_dX[num_nodes * 3];
    dN_dX[0*3+0] = -a; dN_dX[0*3+1] = -a; dN_dX[0*3+2] = -a;
    dN_dX[1*3+0] =  a; dN_dX[1*3+1] = -a; dN_dX[1*3+2] = -a;
    dN_dX[2*3+0] =  a; dN_dX[2*3+1] =  a; dN_dX[2*3+2] = -a;
    dN_dX[3*3+0] = -a; dN_dX[3*3+1] =  a; dN_dX[3*3+2] = -a;
    dN_dX[4*3+0] = -a; dN_dX[4*3+1] = -a; dN_dX[4*3+2] =  a;
    dN_dX[5*3+0] =  a; dN_dX[5*3+1] = -a; dN_dX[5*3+2] =  a;
    dN_dX[6*3+0] =  a; dN_dX[6*3+1] =  a; dN_dX[6*3+2] =  a;
    dN_dX[7*3+0] = -a; dN_dX[7*3+1] =  a; dN_dX[7*3+2] =  a;

    Real F[9];
    compute_deformation_gradient(dN_dX, disp, num_nodes, F);

    // Expected: F = [1.1, 0, 0; 0, 1, 0; 0, 0, 1]
    check(std::abs(F[0] - 1.1) < TOL, "F_11 = 1.1 for 10% x-stretch");
    check(std::abs(F[4] - 1.0) < TOL, "F_22 = 1.0");
    check(std::abs(F[8] - 1.0) < TOL, "F_33 = 1.0");

    Real J = tensor::determinant3x3(F);
    check(std::abs(J - 1.1) < TOL, "J = 1.1 for 10% x-stretch");
}

// Test 4: Green-Lagrange strain
void test_green_lagrange_strain() {
    std::cout << "\n=== Test 4: Green-Lagrange Strain ===\n";

    // F = diag(1.1, 1.0, 1.0) - 10% stretch in x
    Real F[9] = {1.1, 0, 0, 0, 1, 0, 0, 0, 1};

    Real E[6];
    green_lagrange_strain(F, E);

    // E = 0.5*(F^T F - I) = 0.5*([1.21,0,0;0,1,0;0,0,1] - I)
    // E_11 = 0.5*(1.21 - 1) = 0.105
    check(std::abs(E[0] - 0.105) < TOL, "E_11 = 0.105 for 10% stretch");
    check(std::abs(E[1]) < TOL, "E_22 = 0");
    check(std::abs(E[2]) < TOL, "E_33 = 0");
    check(std::abs(E[3]) < TOL, "E_12 = 0 (no shear)");

    // Simple shear test: F = [1, γ, 0; 0, 1, 0; 0, 0, 1]
    Real gamma = 0.2;
    Real F_shear[9] = {1, gamma, 0, 0, 1, 0, 0, 0, 1};

    Real E_shear[6];
    green_lagrange_strain(F_shear, E_shear);

    // C = F^T F = [1, γ; γ, 1+γ²] (2D part)
    // E_11 = 0, E_22 = 0.5*γ², 2*E_12 = γ
    check(std::abs(E_shear[0]) < TOL, "E_11 = 0 for simple shear");
    check(std::abs(E_shear[1] - 0.5*gamma*gamma) < TOL, "E_22 = 0.5*γ² for simple shear");
    check(std::abs(E_shear[3] - gamma) < TOL, "2*E_12 = γ for simple shear");
}

// Test 5: Stress transformations (round-trip)
void test_stress_transformations() {
    std::cout << "\n=== Test 5: Stress Transformations ===\n";

    // Deformation gradient with some stretch and rotation
    Real F[9] = {1.1, 0.05, 0, 0, 0.95, 0.02, 0, 0, 1.0};

    // Original Cauchy stress
    Real sigma[6] = {100e6, 50e6, 30e6, 10e6, 5e6, 2e6};  // Pa

    // Convert to 2nd PK
    Real S[6];
    cauchy_to_second_pk(sigma, F, S);

    // Convert back to Cauchy
    Real sigma_back[6];
    second_pk_to_cauchy(S, F, sigma_back);

    // Should be equal to original
    Real max_err = 0;
    for (int i = 0; i < 6; ++i) {
        Real err = std::abs(sigma_back[i] - sigma[i]) / std::abs(sigma[i] + 1);
        if (err > max_err) max_err = err;
    }
    check(max_err < 1e-10, "Cauchy -> 2nd PK -> Cauchy round-trip");

    std::cout << "  Max relative error: " << max_err << "\n";
}

// Test 6: Rate of deformation tensor
void test_rate_of_deformation() {
    std::cout << "\n=== Test 6: Rate of Deformation ===\n";

    constexpr int num_nodes = 8;

    // Shape function derivatives for element [-1,1]^3
    const Real a = 0.125;
    Real dN_dx[num_nodes * 3];
    dN_dx[0*3+0] = -a; dN_dx[0*3+1] = -a; dN_dx[0*3+2] = -a;
    dN_dx[1*3+0] =  a; dN_dx[1*3+1] = -a; dN_dx[1*3+2] = -a;
    dN_dx[2*3+0] =  a; dN_dx[2*3+1] =  a; dN_dx[2*3+2] = -a;
    dN_dx[3*3+0] = -a; dN_dx[3*3+1] =  a; dN_dx[3*3+2] = -a;
    dN_dx[4*3+0] = -a; dN_dx[4*3+1] = -a; dN_dx[4*3+2] =  a;
    dN_dx[5*3+0] =  a; dN_dx[5*3+1] = -a; dN_dx[5*3+2] =  a;
    dN_dx[6*3+0] =  a; dN_dx[6*3+1] =  a; dN_dx[6*3+2] =  a;
    dN_dx[7*3+0] = -a; dN_dx[7*3+1] =  a; dN_dx[7*3+2] =  a;

    // Unit cube nodes [-1,1]^3 (matches natural coordinates)
    Real coords[num_nodes * 3] = {
        -1, -1, -1,  1, -1, -1,  1, 1, -1,  -1, 1, -1,
        -1, -1,  1,  1, -1,  1,  1, 1,  1,  -1, 1,  1
    };

    // Uniform stretch rate: v_x = rate * x
    Real rate = 100.0;  // 1/s
    Real velocity[num_nodes * 3] = {0};
    for (int n = 0; n < num_nodes; ++n) {
        velocity[n*3 + 0] = rate * coords[n*3 + 0];
    }

    Real D[6];
    rate_of_deformation(dN_dx, velocity, num_nodes, D);

    // D_11 should be = rate
    check(std::abs(D[0] - rate) < TOL * rate, "D_11 = stretch rate");
    check(std::abs(D[1]) < TOL * rate, "D_22 = 0 for x-stretch");
    check(std::abs(D[2]) < TOL * rate, "D_33 = 0 for x-stretch");

    std::cout << "  D = [" << D[0] << ", " << D[1] << ", " << D[2] << ", "
              << D[3] << ", " << D[4] << ", " << D[5] << "]\n";
}

// Test 7: Internal force computation
void test_internal_forces() {
    std::cout << "\n=== Test 7: Internal Force Computation ===\n";

    constexpr int num_nodes = 8;

    // Shape function derivatives
    const Real a = 0.125;
    Real dN_dX[num_nodes * 3];
    dN_dX[0*3+0] = -a; dN_dX[0*3+1] = -a; dN_dX[0*3+2] = -a;
    dN_dX[1*3+0] =  a; dN_dX[1*3+1] = -a; dN_dX[1*3+2] = -a;
    dN_dX[2*3+0] =  a; dN_dX[2*3+1] =  a; dN_dX[2*3+2] = -a;
    dN_dX[3*3+0] = -a; dN_dX[3*3+1] =  a; dN_dX[3*3+2] = -a;
    dN_dX[4*3+0] = -a; dN_dX[4*3+1] = -a; dN_dX[4*3+2] =  a;
    dN_dX[5*3+0] =  a; dN_dX[5*3+1] = -a; dN_dX[5*3+2] =  a;
    dN_dX[6*3+0] =  a; dN_dX[6*3+1] =  a; dN_dX[6*3+2] =  a;
    dN_dX[7*3+0] = -a; dN_dX[7*3+1] =  a; dN_dX[7*3+2] =  a;

    // Uniform stress state
    Real sigma[6] = {1e6, 0, 0, 0, 0, 0};  // 1 MPa in x-direction

    Real f_int[num_nodes * 3] = {0};
    Real detj = 1.0;  // Unit cube
    Real weight = 8.0;  // Total integration weight for 2x2x2 Gauss

    internal_force_updated_lagrangian(dN_dX, sigma, detj, weight, num_nodes, f_int);

    // For uniform stress σ_xx = 1 MPa on unit cube:
    // f_x at nodes with x=1: positive force (tension pulling out)
    // f_x at nodes with x=0: negative force (tension pulling in)

    // Check force equilibrium: sum should be zero
    Real sum_fx = 0, sum_fy = 0, sum_fz = 0;
    for (int n = 0; n < num_nodes; ++n) {
        sum_fx += f_int[n*3 + 0];
        sum_fy += f_int[n*3 + 1];
        sum_fz += f_int[n*3 + 2];
    }

    check(std::abs(sum_fx) < TOL * 1e6, "Force equilibrium in x");
    check(std::abs(sum_fy) < TOL * 1e6, "Force equilibrium in y");
    check(std::abs(sum_fz) < TOL * 1e6, "Force equilibrium in z");

    // Check that forces on x=1 face are positive
    Real fx_x1 = f_int[1*3+0] + f_int[2*3+0] + f_int[5*3+0] + f_int[6*3+0];
    check(fx_x1 > 0, "Tension force on x=1 face is positive");

    std::cout << "  Sum(fx) = " << sum_fx << ", Sum(fy) = " << sum_fy
              << ", Sum(fz) = " << sum_fz << "\n";
}

// Test 8: Geometry update
void test_geometry_update() {
    std::cout << "\n=== Test 8: Geometry Update ===\n";

    constexpr int num_nodes = 4;
    Real coords[num_nodes * 3] = {
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0
    };

    Real delta_disp[num_nodes * 3] = {
        0.1, 0, 0,
        0.1, 0, 0,
        0.1, 0, 0,
        0.1, 0, 0
    };

    Real coords_new[num_nodes * 3];
    update_geometry(coords, delta_disp, num_nodes, coords_new);

    check(std::abs(coords_new[0] - 0.1) < TOL, "Node 0 x-coord updated");
    check(std::abs(coords_new[3] - 1.1) < TOL, "Node 1 x-coord updated");
    check(std::abs(coords_new[6] - 1.1) < TOL, "Node 2 x-coord updated");
    check(std::abs(coords_new[9] - 0.1) < TOL, "Node 3 x-coord updated");
}

// Test 9: Element validity check
void test_element_validity() {
    std::cout << "\n=== Test 9: Element Validity Check ===\n";

    // Valid deformation (10% stretch)
    Real F_valid[9] = {1.1, 0, 0, 0, 1, 0, 0, 0, 1};
    check(check_element_valid(F_valid, 0.01), "10% stretch is valid");

    // Inverted element (negative J)
    Real F_inverted[9] = {-1, 0, 0, 0, 1, 0, 0, 0, 1};
    check(!check_element_valid(F_inverted, 0.01), "Inverted element detected");

    // Severely compressed (J < 0.01)
    Real F_compressed[9] = {0.005, 0, 0, 0, 1, 0, 0, 0, 1};
    check(!check_element_valid(F_compressed, 0.01), "Severely compressed element detected");
}

// Test 10: Neo-Hookean material with large deformation
void test_neohookean_large_deform() {
    std::cout << "\n=== Test 10: Neo-Hookean with Large Deformation ===\n";

    // Material properties (rubber-like)
    MaterialProperties props;
    props.E = 1.0e6;   // 1 MPa
    props.nu = 0.49;   // Nearly incompressible
    props.compute_derived();

    NeoHookeanMaterial neo(props);

    // Large stretch: F = diag(2, 0.7071, 0.7071) - volume preserving
    // J = 2 * 0.7071 * 0.7071 ≈ 1
    MaterialState state;
    Real stretch = 2.0;
    Real lat = 1.0 / std::sqrt(stretch);  // Lateral contraction for incompressibility
    state.F[0] = stretch; state.F[1] = 0; state.F[2] = 0;
    state.F[3] = 0; state.F[4] = lat; state.F[5] = 0;
    state.F[6] = 0; state.F[7] = 0; state.F[8] = lat;

    neo.compute_stress(state);

    Real J = tensor::determinant3x3(state.F);
    std::cout << "  J = " << J << " (should be ~1 for incompressible)\n";
    std::cout << "  σ_xx = " << state.stress[0] / 1e6 << " MPa\n";
    std::cout << "  σ_yy = " << state.stress[1] / 1e6 << " MPa\n";
    std::cout << "  σ_zz = " << state.stress[2] / 1e6 << " MPa\n";

    // For uniaxial tension, σ_yy = σ_zz should be ≈ 0 (free lateral faces)
    // and σ_xx should be positive (tension)
    check(state.stress[0] > 0, "σ_xx > 0 for tensile stretch");
    check(std::abs(J - 1.0) < 0.01, "Volume approximately preserved");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Large Deformation Mechanics Tests\n";
    std::cout << "========================================\n";

    Kokkos::initialize();
    {
        test_tensor_utilities();
        test_deformation_gradient_identity();
        test_deformation_gradient_stretch();
        test_green_lagrange_strain();
        test_stress_transformations();
        test_rate_of_deformation();
        test_internal_forces();
        test_geometry_update();
        test_element_validity();
        test_neohookean_large_deform();
    }
    Kokkos::finalize();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_total << " tests passed\n";
    std::cout << "========================================\n";

    return (tests_passed == tests_total) ? 0 : 1;
}
