# Material Model Catalog

Complete catalog of all 115 NexusSim material models and 20 equations of state
with OpenRadioss LAW ID cross-reference.

---

## Basic Metals (Wave 1)

| NexusSim Class             | LAW   | Description                              | Wave |
|----------------------------|-------|------------------------------------------|------|
| `ElasticMaterial`          | LAW1  | Linear isotropic elastic                 | 1    |
| `OrthotropicMaterial`      | LAW12 | Orthotropic elastic                      | 1    |
| `ElasticPlasticFail`       | LAW2  | Isotropic elastic-plastic with failure   | 1    |
| `JohnsonCookMaterial`      | LAW2  | Johnson-Cook rate/temperature dependent  | 1    |
| `CowperSymondsMaterial`    | LAW44 | Cowper-Symonds rate-dependent             | 1    |
| `PiecewiseLinearMaterial`  | LAW36 | Piecewise-linear plasticity              | 1    |
| `TabulatedMaterial`        | LAW36 | Tabulated stress-strain                  | 1    |
| `RigidMaterial`            | LAW0  | Rigid body material                      | 1    |
| `NullMaterial`             | LAW0  | Null material (zero stiffness)           | 1    |

## Rate-Dependent Metals (Wave 10)

| NexusSim Class                  | LAW   | Description                         | Wave |
|---------------------------------|-------|-------------------------------------|------|
| `HillMaterial`                  | LAW32 | Hill anisotropic yield              | 10   |
| `BarlatMaterial`                | LAW37 | Barlat anisotropic yield (Yld2000) | 10   |
| `TabulatedJCMaterial`           | LAW93 | Tabulated Johnson-Cook              | 10   |
| `ZerilliArmstrongMaterial`     | LAW23 | Zerilli-Armstrong dislocation model | 10   |
| `SteinbergGuinanMaterial`       | LAW17 | Steinberg-Guinan high-rate          | 10   |
| `MTSMaterial`                   | LAW85 | Mechanical threshold stress         | 10   |
| `SpotWeldMaterial`              | LAW59 | Spot weld connection material       | 10   |
| `RateCompositeMaterial`         | LAW58 | Rate-dependent composite            | 10   |
| `ThermalEPMaterial`             | LAW46 | Thermo-elastic-plastic              | 10   |
| `UserDefinedMaterial`           | LAW29 | User-defined interface              | 10   |

## Hyperelastic (Wave 1, 10)

| NexusSim Class               | LAW   | Description                          | Wave |
|------------------------------|-------|--------------------------------------|------|
| `NeoHookeanMaterial`         | LAW42 | Neo-Hookean hyperelastic             | 1    |
| `MooneyRivlinMaterial`       | LAW42 | Mooney-Rivlin 2-parameter            | 1    |
| `OgdenMaterial`              | LAW42 | Ogden N-term hyperelastic            | 1    |
| `ArrudaBoyceMaterial`        | LAW42 | Arruda-Boyce 8-chain                 | 10   |
| `BlatzKoMaterial`            | LAW69 | Blatz-Ko compressible foam           | 10   |
| `SMAMaterial`                | LAW73 | Shape memory alloy (Auricchio)       | 10   |
| `LaminatedGlassMaterial`    | LAW59 | Laminated glass interlayer           | 10   |

## Foam and Crush (Wave 1, 10, 24-25)

| NexusSim Class                  | LAW   | Description                        | Wave |
|---------------------------------|-------|------------------------------------|------|
| `FoamMaterial`                  | LAW33 | Low-density foam                   | 1    |
| `CrushableFoamMaterial`         | LAW33 | Crushable foam (volumetric)        | 1    |
| `HoneycombMaterial`             | LAW28 | Orthotropic honeycomb              | 1    |
| `RateFoamMaterial`              | LAW33 | Rate-dependent foam                | 10   |
| `ScaledCrushFoam`               | LAW70 | Scaled crushable foam              | 24   |
| `AnisotropicCrushFoam`          | LAW70 | Anisotropic crush foam             | 18   |
| `DeshpandeFleckFoam`            | LAW70 | Deshpande-Fleck metal foam         | 32   |
| `ViscousFoam`                   | LAW62 | Viscous damping foam               | 32   |
| `DuboisFoam`                    | LAW70 | Dubois tabulated foam              | 32   |

## Viscoelastic (Wave 1, 10, 31)

| NexusSim Class                     | LAW   | Description                     | Wave |
|------------------------------------|-------|---------------------------------|------|
| `ViscoelasticMaterial`             | LAW34 | Generalized Maxwell viscoelastic| 1    |
| `PronyMaterial`                    | LAW34 | Prony series viscoelastic       | 10   |
| `FrequencyViscoelasticMaterial`    | LAW40 | Frequency-domain viscoelastic   | 18   |
| `GeneralizedViscoelasticMaterial`  | LAW76 | Multi-branch Prony              | 18   |
| `KelvinMaxwellMaterial`            | LAW40 | Kelvin-Maxwell 3-element        | 31   |
| `ViscousTabMaterial`               | LAW40 | Tabulated viscous               | 31   |

## Composites (Wave 7, 10, 18, 25, 32, 39)

| NexusSim Class                  | LAW   | Description                        | Wave |
|---------------------------------|-------|------------------------------------|------|
| `CompositePlyMaterial`          | LAW25 | Composite ply stacking             | 7    |
| `CohesiveMaterial`              | LAW59 | Cohesive zone bilinear             | 10   |
| `TabulatedCompositeMaterial`    | LAW58 | Tabulated composite                | 18   |
| `PlyDegradationMaterial`        | LAW25 | Ply degradation progressive        | 18   |
| `CompositeDamageMaterial`       | LAW53 | Progressive composite damage       | 39   |
| `ExtendedRateComposite`         | LAW58 | Extended rate composite            | 25   |
| `EnhancedComposite`             | LAW25 | Enhanced composite failure         | 32   |
| `NXTComposite` (failure)        | LAW25 | NXT composite failure model        | 33   |

## Concrete and Soil (Wave 10, 18, 24-25, 31-32, 39)

| NexusSim Class                  | LAW   | Description                        | Wave |
|---------------------------------|-------|------------------------------------|------|
| `ConcreteMaterial`              | LAW24 | Concrete damage model              | 10   |
| `SoilCapMaterial`               | LAW14 | Soil & cap model                   | 10   |
| `DruckerPragerMaterial`         | LAW10 | Drucker-Prager yield               | 18   |
| `GranularSoilCapMaterial`       | LAW79 | Granular soil with cap hardening   | 24   |
| `MultiSurfaceConcrete`          | LAW78 | Multi-surface concrete             | 24   |
| `DPCapMaterial`                 | LAW81 | Drucker-Prager with cap            | 39   |
| `MohrCoulombMaterial`           | LAW27 | Classic Mohr-Coulomb               | 39   |
| `ExtendedSoilMaterial`          | LAW21 | Extended soil model                | 25   |
| `SoilCrushMaterial`             | LAW14 | Combined soil & crush              | 25   |
| `DruckerPragerExtMaterial`      | LAW10 | Extended Drucker-Prager            | 31   |
| `Concrete3Surface`              | LAW24 | Three-surface concrete             | 31   |
| `CDPM2Concrete`                 | LAW24 | Concrete damage-plasticity model 2 | 32   |
| `JHConcrete`                    | LAW78 | JH concrete model                  | 32   |
| `ConcreteDPMaterial`            | LAW24 | Concrete damage-plasticity         | 25   |

## Thermal and Metallurgy (Wave 14, 18, 24, 39)

| NexusSim Class                     | LAW   | Description                     | Wave |
|------------------------------------|-------|---------------------------------|------|
| `ThermalSolver`                    | --    | Heat conduction solver           | 14   |
| `ViscoplasticThermalMaterial`      | LAW46 | Viscoplastic thermal coupling   | 18   |
| `ThermalMetallurgyMaterial`        | LAW80 | Phase transformation kinetics   | 39   |
| `ThermoplasticPolymer`             | LAW46 | Thermoplastic polymer           | 24   |
| `HanselHotForm`                    | LAW83 | Hansel hot-forming model        | 31   |
| `HenselSpittelMaterial`            | LAW83 | Hensel-Spittel hot working      | 32   |

## Shell and Specialty (Wave 10, 32, 39)

| NexusSim Class              | LAW   | Description                           | Wave |
|-----------------------------|-------|---------------------------------------|------|
| `ElasticShellMaterial`      | LAW4  | Shell-specific elastic-plastic        | 39   |
| `FabricMaterial`            | LAW19 | Fabric / membrane                     | 10   |
| `FabricNLMaterial`          | LAW19 | Nonlinear fabric                      | 32   |
| `AdvancedFabricMaterial`    | LAW19 | Advanced fabric                       | 24   |

## Equation of State Materials (Wave 5, 13, 18, 31)

| NexusSim Class                 | LAW   | Description                        | Wave |
|--------------------------------|-------|------------------------------------|------|
| `IdealGasEOS`                  | --    | Ideal gas p = rho*R*T              | 5    |
| `GruneisenEOS`                 | --    | Mie-Gruneisen shock Hugoniot       | 5    |
| `JWLEOS`                       | --    | JWL explosive products             | 5    |
| `PolynomialEOS`                | --    | Polynomial in compression/tension  | 5    |
| `TabulatedEOS`                 | --    | Tabulated p(rho, e)                | 5    |
| `MurnaghanEOS`                 | --    | Murnaghan isothermal               | 13   |
| `NobleAbelEOS`                 | --    | Noble-Abel covolume gas            | 13   |
| `StiffGasEOS`                  | --    | Stiffened gas                      | 13   |
| `TillotsonEOS`                 | --    | Tillotson multiphase               | 13   |
| `SesameEOS`                    | --    | SESAME tabular                     | 13   |
| `PowderBurnEOS`                | --    | Powder burn reactive               | 13   |
| `CompactionEOS`                | --    | Porous material compaction         | 13   |
| `OsborneEOS`                   | --    | Osborne quadratic                  | 13   |

## Explosive and Reactive (Wave 18, 31)

| NexusSim Class                   | LAW   | Description                       | Wave |
|----------------------------------|-------|-----------------------------------|------|
| `ExplosiveBurnMaterial`          | LAW5  | Programmed burn + beta burn       | 18   |
| `ProgrammedDetonationMaterial`   | LAW5  | Detonation wave tracking          | 18   |
| `LeeTarverReactiveMaterial`      | LAW97 | Lee-Tarver reactive model         | 31   |
| `ExplosiveBurnExtMaterial`       | LAW5  | Extended explosive burn            | 25   |
| `JWLBMaterial`                   | LAW97 | JWL-B afterburn                   | 32   |

## Hardening and Plasticity (Wave 18, 24-25, 31-32)

| NexusSim Class                     | LAW   | Description                     | Wave |
|------------------------------------|-------|---------------------------------|------|
| `KinematicHardeningMaterial`       | LAW36 | Kinematic hardening (Prager)    | 18   |
| `ChabocheMaterial`                 | LAW79 | Chaboche nonlinear kinematic    | 24   |
| `PolynomialHardeningMaterial`      | LAW52 | Polynomial hardening            | 18   |
| `SpecialHardening`                 | LAW52 | Special hardening model         | 25   |
| `OrthotropicPlasticMaterial`       | LAW32 | Orthotropic plasticity          | 18   |
| `Barlat2000Material`               | LAW37 | Barlat Yld2000-2d               | 24   |
| `OrthotropicHill`                  | LAW32 | Orthotropic Hill plasticity     | 32   |
| `VegterYield`                      | LAW37 | Vegter yield criterion          | 32   |

## Spring and Connector (Wave 18, 25, 32)

| NexusSim Class                    | LAW   | Description                      | Wave |
|-----------------------------------|-------|----------------------------------|------|
| `SpringHysteresisMaterial`        | LAW67 | Spring with hysteresis           | 18   |
| `HysteresisSpringExt`            | LAW67 | Extended spring hysteresis       | 25   |
| `SpringGeneralized`              | LAW67 | Generalized spring-beam          | 32   |
| `BondedInterfaceMaterial`         | LAW59 | Bonded interface (cohesive)      | 18   |
| `ARUPAdhesive`                    | LAW59 | ARUP adhesive model              | 32   |
| `AdhesiveJoint` (failure)         | --    | Adhesive joint failure           | 19   |

## Advanced and Specialized (Wave 18, 24-25, 31-32, 39)

| NexusSim Class                     | LAW   | Description                     | Wave |
|------------------------------------|-------|---------------------------------|------|
| `PorousElasticMaterial`            | LAW62 | Porous elastic                  | 18   |
| `PorousBrittleMaterial`            | LAW62 | Porous brittle                  | 18   |
| `BrittleFractureMaterial`          | LAW27 | Brittle fracture                | 18   |
| `CreepMaterial`                    | LAW46 | Creep (Norton power law)        | 18   |
| `UnifiedCreepMaterial`             | LAW46 | Unified creep plasticity        | 24   |
| `PinchingMaterial`                 | LAW58 | Pinching (cyclic degradation)   | 18   |
| `PhaseTransformMaterial`           | LAW80 | Phase transformation SMA        | 18   |
| `NonlinearElasticMaterial`         | LAW44 | Nonlinear elastic (reversible)  | 39   |
| `ZhaoMaterial`                     | LAW48 | Zhao viscoelastic foam          | 1    |
| `OrthotropicBrittleMaterial`       | LAW12 | Orthotropic brittle             | 25   |
| `AdvancedPolymerMaterial`          | LAW42 | Advanced polymer                | 25   |
| `MultiScaleMaterial`               | LAW25 | Multi-scale composite           | 25   |
| `MarlowhyperelasticMaterial`       | LAW42 | Marlow data-driven hyperelastic | 32   |

## Tier 3 Specialized (Wave 31-32)

| NexusSim Class                  | LAW   | Description                        | Wave |
|---------------------------------|-------|------------------------------------|------|
| `SesameTabMaterial`             | LAW51 | SESAME tabular material            | 31   |
| `BiphasicMaterial`              | LAW10 | Biphasic soil/fluid material       | 31   |
| `FluffMaterial`                 | LAW62 | Fluff / airbag fabric filler       | 31   |
| `LESFluidMaterial`              | LAW6  | LES turbulent fluid material       | 31   |
| `MultiMaterialMaterial`         | LAW51 | Multi-material ALE                 | 31   |
| `PlasticTriangleMaterial`       | LAW4  | Triangular shell plastic           | 31   |
| `UgineALZMaterial`              | LAW83 | Ugine ALZ stainless steel          | 31   |
| `CosseratMaterial`              | LAW32 | Cosserat micropolar                | 31   |
| `YuModelMaterial`               | LAW24 | Yu concrete/soil model             | 31   |
| `ModifiedLaDevezeMaterial`      | LAW25 | Modified LaDeveze delamination     | 32   |
| `GranularMaterial`              | LAW14 | Granular model                     | 32   |
| `PaperLightMaterial`            | LAW4  | Paper/light material               | 32   |
| `PPPolymerMaterial`             | LAW42 | Polypropylene polymer              | 32   |
| `DP3Material`                   | LAW10 | Three-invariant Drucker-Prager     | 32   |
| `JCookAluminumMaterial`         | LAW2  | JCook aluminum (calibrated)        | 32   |

---

## Summary Statistics

| Category          | Count | Waves Introduced                        |
|-------------------|-------|-----------------------------------------|
| Basic metals      | 9     | 1                                       |
| Rate-dependent    | 10    | 10                                      |
| Hyperelastic      | 7     | 1, 10                                   |
| Foam/Crush        | 9     | 1, 10, 18, 24, 32                       |
| Viscoelastic      | 6     | 1, 10, 18, 31                           |
| Composite         | 8     | 7, 10, 18, 25, 32, 33, 39              |
| Concrete/Soil     | 14    | 10, 18, 24, 25, 31, 32, 39             |
| Thermal/Metal     | 6     | 14, 18, 24, 31, 32, 39                 |
| Shell/Fabric      | 4     | 10, 24, 32, 39                          |
| EOS               | 20    | 5, 13, 43                               |
| Explosive         | 5     | 18, 25, 31, 32                          |
| Hardening/Plast.  | 8     | 18, 24, 25, 32                          |
| Spring/Connector  | 6     | 18, 19, 25, 32                          |
| Advanced/Special  | 13    | 1, 18, 24, 25, 32, 39                  |
| Tier 3 Special    | 15    | 31, 32                                  |
| **Total**         |**115**|                                         |
