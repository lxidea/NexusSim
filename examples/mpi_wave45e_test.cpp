/**
 * @file mpi_wave45e_test.cpp
 * @brief Wave 45e: Final tuning + MPI scalability validation (9 tests)
 *
 * Tests: AdaptiveHourglassSelector, ContactStabilizationDamper,
 *        ElementTimeStepCalibrator, MPIScalabilityValidator,
 *        ProductionMPIDriver.
 *
 * Run: mpiexec -n 2 ./mpi_wave45e_test
 */

#include <nexussim/parallel/tuning_wave45.hpp>
#include <nexussim/parallel/mpi_wave45.hpp>
#include <cmath>
#include <thread>

using namespace nxs::parallel;

int main(int argc, char** argv)
{
    MPITestHarness mpi(argc, argv);
    MPITestRunner runner(mpi);

    // -----------------------------------------------------------------------
    // Tests 1-2: AdaptiveHourglassSelector
    // -----------------------------------------------------------------------
    runner.add_test("HG selector: regular hex → IHQ1",
    [&](MPIAssert& a, int /*rank*/, int /*size*/) {
        // Perfect unit cube
        Real coords[8][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };
        ElementQuality q = compute_hex8_quality(coords);
        AdaptiveHourglassSelector selector;
        int mode = selector.select_mode(q);
        a.check_all(mode == 1, "regular hex → IHQ1 (viscous)");
        a.check_all(q.aspect_ratio < 2.0, "regular hex aspect ratio < 2");
    });

    runner.add_test("HG selector: distorted hex → IHQ4 or IHQ5",
    [&](MPIAssert& a, int /*rank*/, int /*size*/) {
        // Highly distorted: extreme aspect ratio + skew
        Real coords[8][3] = {
            {0,0,0}, {20,0,0}, {21,1,0}, {0,1,0},
            {0,0,0.5}, {20,0,0.5}, {21,1,0.5}, {0,1,0.5}
        };
        ElementQuality q = compute_hex8_quality(coords);
        AdaptiveHourglassSelector selector;
        int mode = selector.select_mode(q);
        a.check_all(mode >= 4, "distorted hex → IHQ4 or IHQ5");
        a.check_all(q.aspect_ratio > 5.0, "distorted hex aspect ratio > 5");
    });

    // -----------------------------------------------------------------------
    // Tests 3-4: ContactStabilizationDamper
    // -----------------------------------------------------------------------
    runner.add_test("Contact damper: force computation",
    [&](MPIAssert& a, int /*rank*/, int /*size*/) {
        ContactStabilizationDamper damper(0.1);

        Real k = 1.0e6;   // contact stiffness
        Real m = 0.01;     // nodal mass
        Real v = 10.0;     // relative velocity

        Real force = damper.compute_damping_force(k, m, v);
        // F = 0.1 * 2 * sqrt(1e6 * 0.01) * 10 = 0.1 * 2 * 100 * 10 = 200
        a.check_near_all(force, 200.0, 1.0, "damping force ≈ 200");
    });

    runner.add_test("Contact damper: critical ratio",
    [&](MPIAssert& a, int /*rank*/, int /*size*/) {
        ContactStabilizationDamper damper(0.2);

        Real k = 1.0e4;
        Real m = 1.0;
        Real critical = damper.critical_damping(k, m);
        // 2 * sqrt(1e4 * 1) = 2 * 100 = 200
        a.check_near_all(critical, 200.0, 0.01, "critical damping = 200");

        // Batch test
        std::vector<Real> stiff = {1e4, 1e6};
        std::vector<Real> mass  = {1.0, 0.01};
        std::vector<Real> vel   = {5.0, 10.0};
        auto forces = damper.compute_batch(stiff, mass, vel);

        // F[0] = 0.2 * 200 * 5 = 200
        // F[1] = 0.2 * 200 * 10 = 400
        a.check_near_all(forces[0], 200.0, 1.0, "batch force[0] ≈ 200");
        a.check_near_all(forces[1], 400.0, 1.0, "batch force[1] ≈ 400");
    });

    // -----------------------------------------------------------------------
    // Tests 5-6: ElementTimeStepCalibrator
    // -----------------------------------------------------------------------
    runner.add_test("TimeStep calibrator: per-element safety factors",
    [&](MPIAssert& a, int /*rank*/, int /*size*/) {
        using ET = ElementTimeStepCalibrator::ElementType;
        ElementTimeStepCalibrator cal;

        // Hex8: 0.9
        a.check_near_all(cal.safety_factor(ET::Hex8), 0.9, 0.01,
                         "Hex8 safety factor = 0.9");
        // Tet4: 0.6667
        a.check_near_all(cal.safety_factor(ET::Tet4), 0.6667, 0.01,
                         "Tet4 safety factor ≈ 0.667");
        // Shell4: 0.9
        a.check_near_all(cal.safety_factor(ET::Shell4), 0.9, 0.01,
                         "Shell4 safety factor = 0.9");
    });

    runner.add_test("TimeStep calibrator: explosive material scale",
    [&](MPIAssert& a, int /*rank*/, int /*size*/) {
        using ET = ElementTimeStepCalibrator::ElementType;
        using MC = ElementTimeStepCalibrator::MaterialCategory;
        ElementTimeStepCalibrator cal;

        Real dt_std = cal.compute_dt(0.01, 5000.0, ET::Hex8, MC::Standard);
        Real dt_exp = cal.compute_dt(0.01, 5000.0, ET::Hex8, MC::Explosive);
        // dt_std = 0.9 * 0.01 / 5000 = 1.8e-6
        // dt_exp = 0.9 * 0.67 * 0.01 / 5000 = 1.206e-6
        a.check_all(dt_exp < dt_std, "explosive dt < standard dt");
        a.check_near_all(dt_std, 1.8e-6, 1e-8, "standard dt ≈ 1.8e-6");

        // Batch min dt
        std::vector<Real> lens = {0.01, 0.005};
        std::vector<Real> spds = {5000.0, 6000.0};
        std::vector<ET> types = {ET::Hex8, ET::Tet4};
        Real min_dt = cal.compute_min_dt(lens, spds, types);
        // dt[0] = 0.9*0.01/5000 = 1.8e-6
        // dt[1] = 0.6667*0.005/6000 = 5.556e-7
        a.check_all(min_dt < 1e-6, "batch min dt < 1e-6 (tet4 controls)");
    });

    // -----------------------------------------------------------------------
    // Test 7: MPIScalabilityValidator — strong scaling with 2 ranks
    // -----------------------------------------------------------------------
    runner.add_test("Scalability: strong scaling validation",
    [&](MPIAssert& a, int rank, int size) {
        MPIScalabilityValidator validator(0.5);  // 50% overhead tolerance

        // Simulated workload: computation scales with problem, comm is fixed
        auto workload = [&](Index problem_size) -> std::pair<double, double> {
            // Simulate computation proportional to local work
            double comp = 0.001 * static_cast<double>(problem_size) / size;
            // Simulate communication (fixed overhead)
            double comm = 0.0001;
            return {comp, comm};
        };

        auto result = validator.validate_strong_scaling(workload, 1000, rank, size);

        a.check_all(result.total_time >= 0.0, "total time non-negative");

        // Validate communication overhead
        auto comm_result = validator.validate_communication_overhead(
            0.01, 0.001, rank, size);
        a.check_all(comm_result.passed, "comm overhead < 50% threshold");
        a.check_near_all(comm_result.comm_overhead, 0.1, 0.01,
                         "comm/comp ratio ≈ 0.1");
    });

    // -----------------------------------------------------------------------
    // Tests 8-9: ProductionMPIDriver — full cycle integration
    // -----------------------------------------------------------------------
    runner.add_test("Production driver: cycle completion",
    [&](MPIAssert& a, int rank, int size) {
        ProductionMPIDriver::Config config;
        config.end_time = 1.0e-4;
        config.initial_dt = 1.0e-6;
        config.max_steps = 200;

        ProductionMPIDriver driver(config);

        auto force_func = [&](int /*step*/, Real /*dt*/) -> std::pair<Real, Real> {
            Real ke = 100.0 / size;  // split evenly
            Real ie = 50.0 / size;
            return {ke, ie};
        };

        auto result = driver.run(10, 4, force_func, rank, size);

        a.check_all(result.completed, "cycle reached end_time");
        a.check_all(result.num_steps > 0, "cycle took > 0 steps");
        a.check_near_all(result.total_energy, 150.0, 1.0,
                         "global energy = 100+50 = 150");
    });

    runner.add_test("Production driver: dt synchronization",
    [&](MPIAssert& a, int rank, int size) {
        ProductionMPIDriver::Config config;
        config.end_time = 5.0e-6;
        config.initial_dt = 1.0e-6;
        config.max_steps = 100;

        ProductionMPIDriver driver(config);

        // Each rank uses same dt, verify min_dt_used
        auto force_func = [](int, Real) -> std::pair<Real, Real> {
            return {1.0, 1.0};
        };

        auto result = driver.run(5, 2, force_func, rank, size);

        a.check_all(result.num_steps >= 5, "at least 5 steps taken");
        a.check_near_all(result.min_dt_used, 1.0e-6, 1e-10,
                         "min dt used = initial dt");
    });

    return runner.run_all();
}
