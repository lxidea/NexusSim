# =============================================================================
# NexusSim Test Targets
# Extracted from root CMakeLists.txt for maintainability.
# All test executables defined here are registered with CTest.
# =============================================================================

# Helper macro to reduce boilerplate
macro(nexussim_add_test target_name source_file)
    add_executable(${target_name} examples/${source_file})
    target_link_libraries(${target_name} PRIVATE nexussim)
    add_test(NAME ${target_name} COMMAND ${target_name})
endmacro()

# -----------------------------------------------------------------------------
# Element & Solver Debug Tests
# -----------------------------------------------------------------------------
nexussim_add_test(hex20_force_direction_test hex20_force_direction_test.cpp)
nexussim_add_test(hex8_force_direction_test hex8_force_direction_test.cpp)
nexussim_add_test(hex20_mass_matrix_check hex20_mass_matrix_check.cpp)
nexussim_add_test(hex20_static_load_test hex20_static_load_test.cpp)
nexussim_add_test(hex20_single_step_debug hex20_single_step_debug.cpp)
nexussim_add_test(hex20_multistep_debug hex20_multistep_debug.cpp)
nexussim_add_test(hex20_debug_test hex20_debug_test.cpp)
nexussim_add_test(vonmises_plasticity_test vonmises_plasticity_test.cpp)
nexussim_add_test(johnson_cook_test johnson_cook_test.cpp)
nexussim_add_test(neohookean_test neohookean_test.cpp)
nexussim_add_test(implicit_newmark_test implicit_newmark_test.cpp)
nexussim_add_test(contact_test contact_test.cpp)
nexussim_add_test(vtk_animation_demo vtk_animation_demo.cpp)
nexussim_add_test(adaptive_timestep_test adaptive_timestep_test.cpp)
nexussim_add_test(large_deformation_test large_deformation_test.cpp)
nexussim_add_test(element_erosion_test element_erosion_test.cpp)
nexussim_add_test(sparse_solver_test sparse_solver_test.cpp)
nexussim_add_test(gpu_sparse_test gpu_sparse_test.cpp)
nexussim_add_test(node_to_surface_contact_test node_to_surface_contact_test.cpp)
nexussim_add_test(mesh_partition_test mesh_partition_test.cpp)
nexussim_add_test(hex20_derivative_verify hex20_derivative_verify.cpp)
nexussim_add_test(hex20_force_compare hex20_force_compare.cpp)
nexussim_add_test(hex20_dynamic_debug hex20_dynamic_debug.cpp)
nexussim_add_test(hex20_stiffness_check hex20_stiffness_check.cpp)

# New element tests (Phase 3A)
nexussim_add_test(shell3_element_test shell3_element_test.cpp)
nexussim_add_test(truss_element_test truss_element_test.cpp)
nexussim_add_test(spring_damper_test spring_damper_test.cpp)
nexussim_add_test(radioss_reader_test radioss_reader_test.cpp)

# Phase 3B tests
nexussim_add_test(thermal_coupling_test thermal_coupling_test.cpp)
nexussim_add_test(time_integration_test time_integration_test.cpp)

# Phase 3C tests (SPH)
nexussim_add_test(sph_solver_test sph_solver_test.cpp)
nexussim_add_test(fem_sph_coupling_test fem_sph_coupling_test.cpp)

# -----------------------------------------------------------------------------
# Implicit Solver Tests
# -----------------------------------------------------------------------------
nexussim_add_test(implicit_solver_test implicit_solver_test.cpp)
nexussim_add_test(fem_static_test fem_static_test.cpp)
nexussim_add_test(implicit_dynamic_test implicit_dynamic_test.cpp)
set_tests_properties(implicit_dynamic_test PROPERTIES LABELS "known_fail")
nexussim_add_test(implicit_validation_test implicit_validation_test.cpp)

# -----------------------------------------------------------------------------
# Wave 4: Peridynamics Integration
# -----------------------------------------------------------------------------
nexussim_add_test(pd_bar_tension_test pd_bar_tension_test.cpp)
nexussim_add_test(pd_validation_test pd_validation_test.cpp)
nexussim_add_test(pd_fem_coupling_test pd_fem_coupling_test.cpp)
nexussim_add_test(fem_pd_integration_test fem_pd_integration_test.cpp)
nexussim_add_test(pd_enhanced_test pd_enhanced_test.cpp)
nexussim_add_test(mpi_partition_test mpi_partition_test.cpp)

# -----------------------------------------------------------------------------
# Wave 5: Performance Benchmarks
# -----------------------------------------------------------------------------
nexussim_add_test(comprehensive_benchmark comprehensive_benchmark.cpp)

# Kokkos-only performance test (links directly to Kokkos, not nexussim)
add_executable(kokkos_performance_test examples/kokkos_performance_test.cpp)
target_link_libraries(kokkos_performance_test PRIVATE Kokkos::kokkos)
add_test(NAME kokkos_performance_test COMMAND kokkos_performance_test)

# -----------------------------------------------------------------------------
# Wave 1-8: Core Capability Tests
# -----------------------------------------------------------------------------
nexussim_add_test(material_models_test material_models_test.cpp)
nexussim_add_test(failure_models_test failure_models_test.cpp)
nexussim_add_test(rigid_body_test rigid_body_test.cpp)
nexussim_add_test(loads_system_test loads_system_test.cpp)
nexussim_add_test(tied_contact_eos_test tied_contact_eos_test.cpp)
nexussim_add_test(restart_output_test restart_output_test.cpp)
nexussim_add_test(composite_layup_test composite_layup_test.cpp)
nexussim_add_test(composite_progressive_test composite_progressive_test.cpp)
nexussim_add_test(sensor_ale_test sensor_ale_test.cpp)
nexussim_add_test(realistic_crash_test realistic_crash_test.cpp)
nexussim_add_test(hertzian_mortar_test hertzian_mortar_test.cpp)
nexussim_add_test(enhanced_output_test enhanced_output_test.cpp)
nexussim_add_test(lsdyna_reader_ext_test lsdyna_reader_ext_test.cpp)

# -----------------------------------------------------------------------------
# Waves 9-17: Gap Closure Tests
# -----------------------------------------------------------------------------
nexussim_add_test(explicit_dynamics_test explicit_dynamics_test.cpp)
nexussim_add_test(material_wave10_test material_wave10_test.cpp)
nexussim_add_test(failure_wave11_test failure_wave11_test.cpp)
nexussim_add_test(contact_wave12_test contact_wave12_test.cpp)
nexussim_add_test(eos_wave13_test eos_wave13_test.cpp)
nexussim_add_test(thermal_wave14_test thermal_wave14_test.cpp)
nexussim_add_test(elements_wave15_test elements_wave15_test.cpp)
nexussim_add_test(advanced_wave16_test advanced_wave16_test.cpp)
nexussim_add_test(mpi_wave17_test mpi_wave17_test.cpp)

# -----------------------------------------------------------------------------
# Waves 18-22: Phase 2 Tests
# -----------------------------------------------------------------------------
nexussim_add_test(material_wave18_test material_wave18_test.cpp)
nexussim_add_test(failure_wave19_test failure_wave19_test.cpp)
nexussim_add_test(sph_wave19_test sph_wave19_test.cpp)
nexussim_add_test(output_wave20_test output_wave20_test.cpp)
nexussim_add_test(elements_wave20_test elements_wave20_test.cpp)
nexussim_add_test(ale_wave21_test ale_wave21_test.cpp)
nexussim_add_test(contact_wave21_test contact_wave21_test.cpp)
nexussim_add_test(reader_wave22_test reader_wave22_test.cpp)
nexussim_add_test(preprocess_wave22_test preprocess_wave22_test.cpp)
nexussim_add_test(solver_wave22_test solver_wave22_test.cpp)
