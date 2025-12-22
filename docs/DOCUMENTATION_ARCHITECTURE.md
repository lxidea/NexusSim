# Documentation Architecture - Visual Guide

**Purpose**: Visual representation of NexusSim documentation structure
**Created**: 2025-11-08

---

## 📐 Documentation Hierarchy

```
NexusSim Project Root
│
├── README.md ──────────────────► Project Introduction & Quick Start
│                                 ├─ Features & capabilities
│                                 ├─ Build instructions
│                                 ├─ Example usage
│                                 └─ Link to detailed docs
│
├── TODO.md ────────────────────► Current Development Priorities
│                                 ├─ Critical tasks (this week)
│                                 ├─ Important tasks (2-4 weeks)
│                                 ├─ Medium priority (1-3 months)
│                                 └─ Links to docs/TODO.md for details
│
├── DEVELOPMENT_REFERENCE.md ──► Feature Planning Guide
│                                 ├─ Element library roadmap
│                                 ├─ Material model specifications
│                                 ├─ Solver development plans
│                                 ├─ Multi-physics architecture
│                                 └─ Implementation guidelines
│
├── DOCUMENTATION_MAP.md ──────► Complete Navigation Guide
│                                 ├─ Document index
│                                 ├─ "I want to..." quick links
│                                 ├─ Categories & organization
│                                 └─ Search strategies
│
└── docs/ ──────────────────────► Detailed Technical Documentation
    │
    ├── README.md ──────────────► This Directory Navigation
    │                             ├─ Document architecture
    │                             ├─ Organization by category
    │                             ├─ Quick navigation
    │                             └─ Complete file list
    │
    ├── [Planning & Status]
    │   ├── TODO.md ─────────────────────► Detailed implementation tasks
    │   ├── PROGRESS_VS_GOALS_ANALYSIS.md ► Comprehensive status report
    │   ├── Development_Roadmap_Status.md ► Phase-by-phase progress
    │   ├── WHATS_LEFT.md ───────────────► Remaining work breakdown
    │   └── ELEMENT_LIBRARY_STATUS.md ───► Element inventory
    │
    ├── [Architecture & Design]
    │   ├── Unified_Architecture_Blueprint.md ────► Design specification
    │   ├── Framework_Architecture_Current_State.md ► Implementation status
    │   ├── Coupling_GPU_Specification.md ────────► GPU coupling design
    │   ├── Architecture_Decisions.md ────────────► Design rationale
    │   └── Element_Integration_Strategies.md ────► Element formulations
    │
    ├── [Implementation Guides]
    │   ├── GETTING_STARTED_NEXT_PHASE.md ──► Contributor onboarding
    │   ├── GPU_Activation_Implementation_Plan.md ► GPU development
    │   ├── YAML_Input_Format.md ───────────► Configuration spec
    │   ├── FSI_Field_Registration.md ──────► Multi-physics design
    │   └── FSI_Prototype_Plan.md ──────────► FSI implementation
    │
    ├── [Bug Tracking & Debug]
    │   ├── KNOWN_ISSUES.md ─────────────────► Active bug database
    │   ├── HEX20_FORCE_SIGN_BUG_ANALYSIS.md ► Root cause analysis
    │   ├── HEX20_DEBUG_SESSION_2025-11-07.md ► Debug session log
    │   └── Bending_Test_Analysis.md ────────► Element comparison
    │
    ├── [Planning & Migration]
    │   ├── Legacy_Migration_Roadmap.md ─────► OpenRadioss migration
    │   ├── Migration_Wave_Assignments.md ───► Development waves
    │   ├── Wave_Resourcing_Status.md ───────► Resource allocation
    │   └── AI_analysis_spec_comparison.md ──► Analysis artifacts
    │
    ├── [Session Archives]
    │   ├── SESSION_SUMMARY_2025-10-30.md ───► First discovery
    │   └── SESSION_SUMMARY_2025-11-07.md ───► Progress update
    │
    └── [Historical Archives]
        ├── HISTORICAL_NOTES.md ─────────────► Consolidated history
        ├── DOCUMENTATION_ARCHITECTURE.md ───► This file
        └── archived_hex20_mass_analysis.cpp ► Code artifacts
```

---

## 🔄 Document Relationships

### User Journey Flow

```
New User
   │
   ├──► README.md ─────────────► Learn about project
   │                             ├─ See features
   │                             ├─ Try examples
   │                             └─ Build from source
   │
   └──► If interested in contributing...
        │
        └──► DEVELOPMENT_REFERENCE.md ─► Understand roadmap
             │                           ├─ Pick a feature area
             │                           └─ See implementation guides
             │
             └──► docs/GETTING_STARTED_NEXT_PHASE.md
                  │                      ├─ Setup environment
                  │                      ├─ Run tests
                  │                      └─ Choose first task
                  │
                  └──► docs/TODO.md ───► Get implementation details
                       │                ├─ File locations
                       │                ├─ Success criteria
                       │                └─ Start coding!
                       │
                       └──► (Refer to architecture docs as needed)
```

### Developer Work Flow

```
Daily Development
   │
   ├──► TODO.md (root) ────────► Quick priorities check
   │    │
   │    └──► docs/TODO.md ─────► Get detailed instructions
   │         │                   ├─ Code locations
   │         │                   ├─ Fix paths
   │         │                   └─ Success criteria
   │         │
   │         ├──► Architecture docs ─► Design reference
   │         │                         └─ Return to implementation
   │         │
   │         ├──► Known Issues ───────► Check for blockers
   │         │                         └─ Return to implementation
   │         │
   │         └──► Element Status ─────► Verify component ready
   │                                   └─ Return to implementation
   │
   └──► Update docs/TODO.md when done
```

### Bug Investigation Flow

```
Bug Discovered
   │
   ├──► docs/KNOWN_ISSUES.md ──────► Check if known
   │    │
   │    ├──► If known ──────────────► Read analysis doc
   │    │                            └─ Follow fix path
   │    │
   │    └──► If new ────────────────► Create debug session doc
   │         │
   │         └──► Investigate ──────► Update session log
   │              │
   │              └──► Root cause ──► Create analysis doc
   │                   │
   │                   └──► Fix ────► Update KNOWN_ISSUES.md
   │                        │
   │                        └──► Archive session to HISTORICAL_NOTES.md
```

### Feature Planning Flow

```
New Feature Idea
   │
   ├──► DEVELOPMENT_REFERENCE.md ──► Check if planned
   │    │
   │    ├──► If planned ─────────────► Review specification
   │    │                             └─ Add to TODO.md
   │    │
   │    └──► If new ─────────────────► Design architecture
   │         │                         ├─ Update DEVELOPMENT_REFERENCE.md
   │         │                         ├─ Create architecture doc
   │         │                         └─ Add detailed tasks to docs/TODO.md
   │         │
   │         └──► Implementation ────► Follow development workflow
   │              │
   │              └──► Complete ─────► Update status docs
   │                   │
   │                   └──► Archive design notes to HISTORICAL_NOTES.md
```

---

## 📊 Document Categories & Purpose

### Level 1: User-Facing (Root Directory)

```
┌─────────────────────────────────────────────────────────┐
│                    ROOT DIRECTORY                       │
│         Quick access, high-level information            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  README.md               Project introduction          │
│  TODO.md                 Current priorities            │
│  DEVELOPMENT_REFERENCE   Feature planning              │
│  DOCUMENTATION_MAP       Complete navigation           │
│                                                         │
│  Purpose: Easy discovery, user orientation             │
│  Audience: All users, new contributors                 │
│  Update: As needed (README), weekly (TODO)             │
└─────────────────────────────────────────────────────────┘
```

### Level 2: Technical Documentation (docs/)

```
┌─────────────────────────────────────────────────────────┐
│                   DOCS DIRECTORY                        │
│        Detailed specifications and analysis             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Planning & Status] ─────► Current state & tasks      │
│  [Architecture]      ─────► Design & specifications    │
│  [Implementation]    ─────► How-to guides              │
│  [Bug Tracking]      ─────► Active investigations      │
│  [Planning]          ─────► Future roadmap             │
│  [Historical]        ─────► Archived context           │
│                                                         │
│  Purpose: Deep technical reference                     │
│  Audience: Developers, architects, contributors        │
│  Update: Daily to monthly depending on category        │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Navigation Patterns

### By Task Type

```
Task Type                 Start Here              Then Go To
─────────────────────────────────────────────────────────────
Quick check priorities    TODO.md (root)          →
Need implementation       TODO.md (root)          → docs/TODO.md
Understand architecture   README.md               → docs/Unified_Architecture_Blueprint.md
Fix a bug                 docs/KNOWN_ISSUES.md    → docs/[Bug Analysis].md
Plan new feature          DEVELOPMENT_REFERENCE   → docs/TODO.md
Check project status      README.md               → docs/PROGRESS_VS_GOALS_ANALYSIS.md
Understand history        DOCUMENTATION_MAP       → docs/HISTORICAL_NOTES.md
```

### By Role

```
Role                 Entry Point                  Common Docs
───────────────────────────────────────────────────────────────
User                 README.md                    Examples, Quick Start
New Contributor      README.md                    → docs/GETTING_STARTED_NEXT_PHASE.md
Active Developer     TODO.md                      → docs/TODO.md, Known Issues
Architect            DEVELOPMENT_REFERENCE        → docs/Architecture docs
Project Manager      README.md                    → docs/PROGRESS_VS_GOALS_ANALYSIS.md
Stakeholder          README.md                    → docs/Development_Roadmap_Status.md
```

### By Component

```
Component            Reference Docs                               Status Docs
──────────────────────────────────────────────────────────────────────────────
Elements             DEVELOPMENT_REFERENCE                        → docs/ELEMENT_LIBRARY_STATUS.md
                     docs/Element_Integration_Strategies.md

Materials            DEVELOPMENT_REFERENCE                        → docs/TODO.md
                     (Material Models section)

Solvers              DEVELOPMENT_REFERENCE                        → docs/Development_Roadmap_Status.md
                     docs/Unified_Architecture_Blueprint.md

GPU                  docs/GPU_Activation_Implementation_Plan.md  → docs/PROGRESS_VS_GOALS_ANALYSIS.md
                     docs/Coupling_GPU_Specification.md

Multi-Physics        docs/FSI_Field_Registration.md              → docs/Development_Roadmap_Status.md
                     docs/FSI_Prototype_Plan.md
```

---

## 🔍 Quick Reference Guide

### "Where do I find information about..."

| Topic | Primary Document | Secondary References |
|-------|-----------------|---------------------|
| **Current tasks** | `TODO.md` (root) | `docs/TODO.md` |
| **Project status** | `docs/PROGRESS_VS_GOALS_ANALYSIS.md` | `README.md`, `docs/Development_Roadmap_Status.md` |
| **Element library** | `docs/ELEMENT_LIBRARY_STATUS.md` | `DEVELOPMENT_REFERENCE.md` |
| **Known bugs** | `docs/KNOWN_ISSUES.md` | `docs/HEX20_FORCE_SIGN_BUG_ANALYSIS.md` |
| **Architecture** | `docs/Unified_Architecture_Blueprint.md` | `docs/Framework_Architecture_Current_State.md` |
| **GPU implementation** | `docs/GPU_Activation_Implementation_Plan.md` | `docs/Coupling_GPU_Specification.md` |
| **Feature planning** | `DEVELOPMENT_REFERENCE.md` | `docs/Development_Roadmap_Status.md` |
| **Getting started** | `README.md` | `docs/GETTING_STARTED_NEXT_PHASE.md` |
| **History** | `docs/HISTORICAL_NOTES.md` | `docs/SESSION_SUMMARY_*.md` |
| **Documentation structure** | `DOCUMENTATION_MAP.md` | `docs/README.md`, This file |

---

## 📈 Document Update Frequency

```
Update Frequency        Documents                               Who Updates
────────────────────────────────────────────────────────────────────────────
Daily                   TODO.md (root)                          Active developers
                        docs/TODO.md
                        docs/KNOWN_ISSUES.md

Weekly                  docs/TODO.md (task status)              Team lead
                        Session summaries (if sessions)

Monthly                 README.md (roadmap)                     Project manager
                        docs/PROGRESS_VS_GOALS_ANALYSIS.md
                        docs/ELEMENT_LIBRARY_STATUS.md
                        docs/Development_Roadmap_Status.md

Quarterly               DEVELOPMENT_REFERENCE.md                Architects
                        Architecture documents

As Needed               Debug analysis docs                     Investigators
                        Implementation guides                   Feature developers
                        Session summaries                       Session participants

Rarely/Never            docs/Unified_Architecture_Blueprint.md  Architects (major changes)
                        docs/Element_Integration_Strategies.md  (Stable reference)
```

---

## 🔄 Document Lifecycle States

```
State          Description                  Examples                      Action
─────────────────────────────────────────────────────────────────────────────
ACTIVE         Updated regularly            TODO.md                       Update frequently
                                            KNOWN_ISSUES.md
                                            PROGRESS_VS_GOALS_ANALYSIS.md

STABLE         Reference only, rarely       Unified_Architecture_         Preserve, rarely update
               changed                      Blueprint.md
                                            Element_Integration_
                                            Strategies.md

ARCHIVED       Historical reference,        SESSION_SUMMARY_2025-10-30    Move to HISTORICAL_NOTES.md
               no updates                   Debug analyses (resolved)      after 6-12 months

DEPRECATED     No longer relevant,          (None currently)              Remove or clearly mark
               superseded                                                  as deprecated
```

---

## 🎨 Visual Documentation Map

```
                                    NexusSim Documentation
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
              [Root Docs]             [docs/ Folder]          [Code Comments]
                    │                       │                       │
        ┌───────────┼───────────┐          │                   [Doxygen]
        │           │           │           │                   [Inline docs]
    README.md   TODO.md    DEV_REF.md      │
        │           │           │           │
        │           │           │           │
    Users      Developers   Architects      │
        │           │           │           │
        │           └───────────┼───────────┘
        │                       │
        │               ┌───────┴───────┐
        │               │               │
        │         [By Category]    [By Lifecycle]
        │               │               │
        │       ┌───────┼───────┐       │
        │       │       │       │       │
        │    Status  Arch   Impl      Active
        │       │       │       │       │
        │       │       │       │    Stable
        │       │       │       │       │
        │       │       │       │   Archived
        │       │       │       │
        └───────┴───────┴───────┴───────┘
                    │
            [Linked Navigation]
                    │
        ┌───────────┼───────────┐
        │           │           │
  DOCUMENTATION_MAP.md      docs/README.md
        │           │           │
        │           └───────────┘
        │                 │
    [User Journey]    [Developer Journey]
        │                 │
        └─────────────────┘
                │
        [Find Information]
```

---

## 📋 Checklist for Adding New Documentation

### Before Creating New Document

- [ ] Check if information belongs in existing document
- [ ] Determine category (Status, Architecture, Implementation, Debug, Planning, Historical)
- [ ] Decide location (root for user-facing, docs/ for technical)
- [ ] Choose appropriate name (follow naming conventions)
- [ ] Identify related documents for cross-referencing

### Document Content

- [ ] Include header with title, purpose, date
- [ ] Add table of contents (if >3 sections)
- [ ] Use clear section headings
- [ ] Include code examples where helpful
- [ ] Add cross-references to related docs
- [ ] Include file paths and line numbers
- [ ] Add footer with version/date/maintainer

### After Creating Document

- [ ] Update `docs/README.md` (add to appropriate category)
- [ ] Update `DOCUMENTATION_MAP.md` (add to index and quick links)
- [ ] Update root `README.md` if user-facing
- [ ] Add cross-references from related documents
- [ ] Test all internal links
- [ ] Commit with clear message

---

## 🎓 Best Practices

### For Document Authors

1. **Write for your audience**:
   - Root docs → Users and new contributors
   - docs/ → Developers and technical staff

2. **Keep documents focused**:
   - One purpose per document
   - Split if growing too large (>1000 lines)

3. **Cross-reference liberally**:
   - Link to related documents
   - Avoid duplicating information

4. **Update regularly**:
   - Keep status docs current
   - Archive outdated information

5. **Use consistent formatting**:
   - Follow markdown style guide
   - Use tables for comparisons
   - Include code examples

### For Document Readers

1. **Start with navigation docs**:
   - `DOCUMENTATION_MAP.md` for overview
   - `docs/README.md` for technical details

2. **Follow the links**:
   - Documents are heavily cross-referenced
   - Trust the navigation structure

3. **Check last updated date**:
   - Prefer recent documents for current status
   - Use archived docs for historical context

4. **Search strategically**:
   - By task: "I want to..."
   - By component: Element, Material, Solver
   - By role: User, Developer, Architect

---

*This document provides a visual and conceptual map of NexusSim documentation*
*For detailed navigation, see: `../DOCUMENTATION_MAP.md`*
*For docs/ directory contents, see: `README.md`*
*Created: 2025-11-08*
*Maintainer: NexusSim Development Team*
