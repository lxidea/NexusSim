# NexusSim Documentation Map

**Purpose**: Quick reference guide to all project documentation
**Last Updated**: 2025-12-28

---

## 📂 Root Directory Documents

### Essential Files (Read These First!)

1. **`README.md`** - Project overview and quick start
   - Project description
   - Features and capabilities
   - Build instructions
   - Quick start examples
   - Roadmap summary

2. **`docs/PROJECT_CONTEXT.md`** - **START HERE after crash/disconnect**
   - Complete project ecosystem (OpenRadioss, PeriSys-Haoran)
   - Project vision and goals
   - Current implementation status
   - Session recovery checklist
   - Reference to parent folder structure

3. **`TODO.md`** - Current development priorities
   - Critical tasks (this week)
   - Important tasks (2-4 weeks)
   - Medium priority (1-3 months)
   - Quick reference checklist

4. **`DEVELOPMENT_REFERENCE.md`** - Long-term feature planning
   - Feature categories and roadmap
   - Element library expansion plans
   - Material model specifications
   - Solver development guide
   - Implementation guidelines
   - 5-10 year vision

---

## 📚 docs/ Directory Structure

### Planning & Architecture (28 documents total)

#### Project Context (READ FIRST)
- **`PROJECT_CONTEXT.md`** - **Complete project ecosystem reference**
  - OpenRadioss and PeriSys-Haoran integration
  - Project vision, goals, and current status
  - Session recovery checklist

#### Core Architecture
- **`Unified_Architecture_Blueprint.md`** - Overall system design
- **`Framework_Architecture_Current_State.md`** - Implementation status
- **`Coupling_GPU_Specification.md`** - GPU coupling design

#### Development Status
- **`TODO.md`** - Detailed task list with implementation notes
- **`PROGRESS_VS_GOALS_ANALYSIS.md`** - Comprehensive progress analysis
- **`WHATS_LEFT.md`** - Remaining work breakdown (Wave 3 + Wave 4)
- **`Development_Roadmap_Status.md`** - Phase-by-phase status
- **`GETTING_STARTED_NEXT_PHASE.md`** - Contributor quick start

#### Element Library
- **`ELEMENT_LIBRARY_STATUS.md`** - All 7 elements detailed status
- **`Element_Integration_Strategies.md`** - Integration scheme theory

#### Known Issues & Debugging
- **`KNOWN_ISSUES.md`** - Active bug tracking
- **`HEX20_FORCE_SIGN_BUG_ANALYSIS.md`** - Root cause analysis
- **`HEX20_DEBUG_SESSION_2025-11-07.md`** - Debug session log
- **`Bending_Test_Analysis.md`** - Hex8 vs Hex20 comparison

#### Session Summaries
- **`SESSION_SUMMARY_2025-10-30.md`** - First code discovery
- **`SESSION_SUMMARY_2025-11-07.md`** - Major progress update

#### Specifications & Design
- **`YAML_Input_Format.md`** - Configuration file spec
- **`FSI_Field_Registration.md`** - Multi-physics field exchange
- **`FSI_Prototype_Plan.md`** - Fluid-structure interaction plan
- **`Legacy_Migration_Roadmap.md`** - OpenRadioss migration
- **`GPU_Activation_Implementation_Plan.md`** - GPU development guide

#### Historical Context
- **`HISTORICAL_NOTES.md`** - Development history archive (Nov 8)
- **`archived_hex20_mass_analysis.cpp`** - Mass matrix validation code

#### Analysis & Comparison
- **`AI_analysis_spec_comparison.md`** - Spec vs implementation
- **`Architecture_Decisions.md`** - Design choices log
- **`Migration_Wave_Assignments.md`** - Development wave planning
- **`Wave_Resourcing_Status.md`** - Resource allocation

---

## 🗺️ Documentation Navigation

### "I want to..."

**...get started quickly**
→ `README.md` → `TODO.md`

**...contribute to development**
→ `DEVELOPMENT_REFERENCE.md` → `docs/GETTING_STARTED_NEXT_PHASE.md` → `docs/TODO.md`

**...understand the architecture**
→ `docs/Unified_Architecture_Blueprint.md` → `docs/Framework_Architecture_Current_State.md`

**...see what's working**
→ `docs/ELEMENT_LIBRARY_STATUS.md` → `docs/PROGRESS_VS_GOALS_ANALYSIS.md`

**...know what's broken**
→ `docs/KNOWN_ISSUES.md` → `docs/HEX20_FORCE_SIGN_BUG_ANALYSIS.md`

**...plan a new feature**
→ `DEVELOPMENT_REFERENCE.md` → `docs/TODO.md`

**...understand the history**
→ `docs/HISTORICAL_NOTES.md` → `docs/SESSION_SUMMARY_*.md`

**...add a new element**
→ `DEVELOPMENT_REFERENCE.md` (Section: Adding New Element Type)

**...add a new material**
→ `DEVELOPMENT_REFERENCE.md` (Section: Adding New Material Model)

**...fix the Hex20 bug**
→ `docs/HEX20_FORCE_SIGN_BUG_ANALYSIS.md` → `TODO.md`

**...run GPU benchmarks**
→ `docs/GPU_Activation_Implementation_Plan.md` → `TODO.md`

---

## 📋 Document Categories

### By Purpose

**User Documentation**:
- `README.md` - Project introduction
- Examples in `examples/` directory

**Developer Documentation**:
- `DEVELOPMENT_REFERENCE.md` - Feature planning
- `docs/GETTING_STARTED_NEXT_PHASE.md` - Quick start
- `docs/TODO.md` - Implementation tasks

**Architecture Documentation**:
- `docs/Unified_Architecture_Blueprint.md` - Design
- `docs/Framework_Architecture_Current_State.md` - Implementation
- `docs/Coupling_GPU_Specification.md` - GPU design

**Status Documentation**:
- `TODO.md` - Current priorities
- `docs/PROGRESS_VS_GOALS_ANALYSIS.md` - Progress report
- `docs/ELEMENT_LIBRARY_STATUS.md` - Element status
- `docs/KNOWN_ISSUES.md` - Bug tracking

**Historical Documentation**:
- `docs/HISTORICAL_NOTES.md` - Development archive
- `docs/SESSION_SUMMARY_*.md` - Session logs
- Debug analysis documents

### By Recency

**Most Recent** (2025-11-08):
- `TODO.md`
- `DEVELOPMENT_REFERENCE.md`
- `docs/HISTORICAL_NOTES.md`
- `docs/PROGRESS_VS_GOALS_ANALYSIS.md`

**Recent** (2025-11-07):
- `docs/HEX20_FORCE_SIGN_BUG_ANALYSIS.md`
- `docs/SESSION_SUMMARY_2025-11-07.md`
- `docs/ELEMENT_LIBRARY_STATUS.md`
- `docs/KNOWN_ISSUES.md`

**Foundational** (2025-10-30 and earlier):
- `README.md`
- `docs/Unified_Architecture_Blueprint.md`
- `docs/Element_Integration_Strategies.md`

---

## 📊 Documentation Statistics

**Total Documents**: 30
- Root directory: 3 (.md files)
- docs/ directory: 27 (.md + 1 .cpp)

**Categories**:
- Architecture & Design: 5
- Status & Progress: 8
- Element Library: 2
- Bug Tracking: 3
- Session Summaries: 2
- Specifications: 5
- Historical: 2
- Planning: 3

**Total Documentation**: ~100,000+ words

---

## 🔄 Documentation Workflow

### Regular Updates

**Daily** (Active Development):
- Update `TODO.md` task status
- Log in session summaries if major discovery

**Weekly** (Sprint Reviews):
- Update `docs/PROGRESS_VS_GOALS_ANALYSIS.md`
- Update `docs/KNOWN_ISSUES.md`
- Archive completed tasks

**Monthly** (Milestone Reviews):
- Update `README.md` roadmap
- Update `docs/Development_Roadmap_Status.md`
- Update `docs/ELEMENT_LIBRARY_STATUS.md`

**As Needed**:
- `DEVELOPMENT_REFERENCE.md` when planning features
- `docs/HISTORICAL_NOTES.md` when archiving artifacts
- Architecture docs when design changes

### Document Lifecycle

```
Plan → Implement → Document → Review → Archive
  ↓         ↓           ↓          ↓        ↓
TODO.md   Code    TODO complete   PR     HISTORICAL
```

---

## 🎯 Quick Reference

### Most Important 6 Documents

1. **`docs/PROJECT_CONTEXT.md`** - **Start here after crash/disconnect**
2. **`README.md`** - Project overview
3. **`TODO.md`** - What to do now
4. **`docs/WHATS_LEFT.md`** - Detailed remaining work
5. **`DEVELOPMENT_REFERENCE.md`** - Where we're going
6. **`docs/KNOWN_ISSUES.md`** - What's broken

### Critical Bug Fix Path

1. `docs/KNOWN_ISSUES.md` - Understand the problem
2. `docs/HEX20_FORCE_SIGN_BUG_ANALYSIS.md` - Root cause analysis
3. `TODO.md` - Fix instructions
4. `src/discretization/fem/solid/hex20.cpp` - The code

### New Contributor Path

1. `README.md` - Understand the project
2. Build and run tests
3. `docs/GETTING_STARTED_NEXT_PHASE.md` - Next steps
4. `DEVELOPMENT_REFERENCE.md` - Pick a feature
5. `docs/TODO.md` - Implementation details

---

## 📝 Maintenance Notes

**Document Owner**: NexusSim Development Team

**Review Schedule**:
- Weekly: TODO.md, KNOWN_ISSUES.md
- Monthly: README.md, PROGRESS_VS_GOALS_ANALYSIS.md
- Quarterly: DEVELOPMENT_REFERENCE.md, Architecture docs

**Archive Policy**:
- Session summaries: Keep last 6 months
- Bug analyses: Keep until resolved + 1 year
- Code artifacts: Archive in docs/ with "archived_" prefix

**Quality Standards**:
- Clear headings and structure
- Tables for comparisons
- Code examples where helpful
- Cross-references to related docs
- Last updated date

---

*Created: 2025-11-08*
*Updated: 2025-12-28*
*Purpose: Navigation guide for all project documentation*
