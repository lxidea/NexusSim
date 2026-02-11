#pragma once

/**
 * @file material_models.hpp
 * @brief Umbrella header for all NexusSim material models
 *
 * Include this single header to get access to all material models.
 * Individual headers can also be included separately for reduced
 * compile times.
 */

// Base material system (Elastic, VonMises, JohnsonCook, NeoHookean)
#include <nexussim/physics/material.hpp>
#include <nexussim/physics/material_library.hpp>

// Wave 1: Expanded material models
#include <nexussim/physics/material_orthotropic.hpp>
#include <nexussim/physics/material_mooney_rivlin.hpp>
#include <nexussim/physics/material_ogden.hpp>
#include <nexussim/physics/material_piecewise_linear.hpp>
#include <nexussim/physics/material_tabulated.hpp>
#include <nexussim/physics/material_foam.hpp>
#include <nexussim/physics/material_crushable_foam.hpp>
#include <nexussim/physics/material_honeycomb.hpp>
#include <nexussim/physics/material_viscoelastic.hpp>
#include <nexussim/physics/material_cowper_symonds.hpp>
#include <nexussim/physics/material_zhao.hpp>
#include <nexussim/physics/material_elastic_plastic_fail.hpp>
#include <nexussim/physics/material_rigid.hpp>
#include <nexussim/physics/material_null.hpp>
