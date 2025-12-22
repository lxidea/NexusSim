# Quick Reference Card - NexusSim Documentation

**Purpose**: Fast lookup for common documentation tasks
**Print-friendly**: Yes (single page reference)

---

## 📂 File Locations

### Root Directory (4 files)
```
README.md                 Project overview
TODO.md                   Current priorities (quick)
DEVELOPMENT_REFERENCE.md  Feature planning
DOCUMENTATION_MAP.md      Complete navigation
```

### docs/ Directory (29 files)
```
README.md                        Directory guide
TODO.md                          Detailed tasks
PROGRESS_VS_GOALS_ANALYSIS.md    Status report
ELEMENT_LIBRARY_STATUS.md        Element status
KNOWN_ISSUES.md                  Bug tracking
HISTORICAL_NOTES.md              Development archive
DOCUMENTATION_ARCHITECTURE.md    Visual guide
(+ 22 other organized documents)
```

---

## 🎯 "I want to..." Quick Links

| Task | Go To |
|------|-------|
| **Start contributing** | `README.md` → `docs/GETTING_STARTED_NEXT_PHASE.md` |
| **See current tasks** | `TODO.md` → `docs/TODO.md` |
| **Check project status** | `docs/PROGRESS_VS_GOALS_ANALYSIS.md` |
| **Fix a bug** | `docs/KNOWN_ISSUES.md` → `docs/HEX20_FORCE_SIGN_BUG_ANALYSIS.md` |
| **Understand architecture** | `docs/Unified_Architecture_Blueprint.md` |
| **Plan a feature** | `DEVELOPMENT_REFERENCE.md` |
| **See what's working** | `docs/ELEMENT_LIBRARY_STATUS.md` |
| **Understand GPU code** | `docs/GPU_Activation_Implementation_Plan.md` |
| **Learn the history** | `docs/HISTORICAL_NOTES.md` |
| **Navigate all docs** | `DOCUMENTATION_MAP.md` |

---

## 🔍 Search Strategy

### By Component
```
Elements   → docs/ELEMENT_LIBRARY_STATUS.md
Materials  → DEVELOPMENT_REFERENCE.md (Materials section)
Solvers    → DEVELOPMENT_REFERENCE.md (Solvers section)
GPU        → docs/GPU_Activation_Implementation_Plan.md
Testing    → docs/ELEMENT_LIBRARY_STATUS.md (test results)
```

### By Activity
```
Coding     → TODO.md → docs/TODO.md
Planning   → DEVELOPMENT_REFERENCE.md
Debugging  → docs/KNOWN_ISSUES.md
Designing  → docs/Unified_Architecture_Blueprint.md
Reviewing  → docs/PROGRESS_VS_GOALS_ANALYSIS.md
```

### By Role
```
User          → README.md
Developer     → TODO.md → docs/TODO.md
Architect     → DEVELOPMENT_REFERENCE.md → docs/Architecture
Manager       → docs/PROGRESS_VS_GOALS_ANALYSIS.md
Stakeholder   → README.md → docs/Development_Roadmap_Status.md
```

---

## 📊 Critical Files (Top 10)

| Priority | File | Purpose |
|----------|------|---------|
| 1 | `README.md` | Project introduction |
| 2 | `TODO.md` | Current tasks (quick) |
| 3 | `docs/TODO.md` | Detailed implementation |
| 4 | `docs/PROGRESS_VS_GOALS_ANALYSIS.md` | Status report |
| 5 | `DEVELOPMENT_REFERENCE.md` | Feature planning |
| 6 | `docs/KNOWN_ISSUES.md` | Bug tracking |
| 7 | `docs/ELEMENT_LIBRARY_STATUS.md` | Element status |
| 8 | `docs/Unified_Architecture_Blueprint.md` | Design spec |
| 9 | `DOCUMENTATION_MAP.md` | Navigation guide |
| 10 | `docs/HISTORICAL_NOTES.md` | Development history |

---

## 🚨 Critical Bug Fix Path

```
1. docs/KNOWN_ISSUES.md
   └─► Find: Hex20 Force Sign Bug

2. docs/HEX20_FORCE_SIGN_BUG_ANALYSIS.md
   └─► Root cause: Shape function derivatives wrong sign

3. TODO.md
   └─► Fix instructions (1-2 hours)

4. src/discretization/fem/solid/hex20.cpp
   └─► Lines 90-180, 312-314
```

---

## 🎓 New Contributor Path

```
1. README.md
   └─► Understand project

2. Build & run tests
   └─► Verify setup

3. docs/GETTING_STARTED_NEXT_PHASE.md
   └─► Choose priority task

4. DEVELOPMENT_REFERENCE.md
   └─► Pick feature area

5. docs/TODO.md
   └─► Get implementation details

6. Start coding!
```

---

## 📋 Update Schedule

| Frequency | Documents | Who |
|-----------|-----------|-----|
| **Daily** | `TODO.md`, `docs/TODO.md`, `docs/KNOWN_ISSUES.md` | Developers |
| **Weekly** | Task status updates | Team lead |
| **Monthly** | `README.md`, `docs/PROGRESS_VS_GOALS_ANALYSIS.md` | Manager |
| **Quarterly** | `DEVELOPMENT_REFERENCE.md`, Architecture docs | Architects |
| **As Needed** | Debug analyses, session summaries | Investigators |

---

## 🔄 Common Workflows

### Daily Development
```
TODO.md → docs/TODO.md → Code → Update status
```

### Bug Investigation
```
KNOWN_ISSUES.md → Debug analysis → Fix → Update docs
```

### Feature Planning
```
DEVELOPMENT_REFERENCE.md → docs/TODO.md → Implement → Status update
```

### Status Report
```
PROGRESS_VS_GOALS_ANALYSIS.md → Element status → Roadmap → Report
```

---

## 📞 Quick Contacts

| Need | Document | Action |
|------|----------|--------|
| **Can't find info** | `DOCUMENTATION_MAP.md` | Complete index |
| **Don't know where to start** | `README.md` | Project intro |
| **Need to report bug** | `docs/KNOWN_ISSUES.md` | Check if known |
| **Want to contribute** | `docs/GETTING_STARTED_NEXT_PHASE.md` | Onboarding |
| **Need architecture info** | `docs/README.md` | Category index |

---

## 📐 Documentation Stats

```
Total Files:        32 (4 root + 28 docs/)
Total Lines:        ~100,000+
Categories:         7 (Status, Architecture, Implementation, Debug, Planning, Session, Historical)
Languages:          Markdown (31), C++ (1 archived)
Update Frequency:   Daily to Quarterly
Cross-References:   Extensive throughout
```

---

## 🎯 One-Page Cheat Sheet

```
┌──────────────────────────────────────────────────────────┐
│                   NEXUSSIM DOCS                          │
├──────────────────────────────────────────────────────────┤
│ New User        → README.md                              │
│ Developer       → TODO.md                                │
│ Bug Fix         → docs/KNOWN_ISSUES.md                   │
│ Feature Plan    → DEVELOPMENT_REFERENCE.md               │
│ Status Check    → docs/PROGRESS_VS_GOALS_ANALYSIS.md     │
│ Architecture    → docs/Unified_Architecture_Blueprint.md │
│ History         → docs/HISTORICAL_NOTES.md               │
│ Navigation      → DOCUMENTATION_MAP.md                   │
├──────────────────────────────────────────────────────────┤
│ ROOT:   4 files (user-facing, quick access)             │
│ DOCS:  28 files (technical, detailed)                   │
│ TOTAL: 32 files (organized by purpose)                  │
└──────────────────────────────────────────────────────────┘
```

---

*Keep this card handy for quick documentation lookups!*
*Last Updated: 2025-11-08*
