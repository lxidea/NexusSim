# NextGen Project Documentation Index

Start here before diving into any specific design or implementation file. This index explains the purpose of each Markdown document and the order to consume them when bootstrapping development.

## Reading Order & Purpose

1. **Unified_Architecture_Blueprint.md** – Primary entry point summarising the overall architecture, layers, module decomposition, and phased implementation plan.
2. **Architecture_Decisions.md** – Captures concrete technology choices, rationales, and resolved questions. Review immediately after the blueprint to understand guardrails.
3. **Legacy_Migration_Roadmap.md** – Describes migration waves and objectives for porting the legacy code.
4. **Migration_Wave_Assignments.md** – Maps roadmap waves to owners, schedules, and checkpoints.
5. **Wave_Resourcing_Status.md** – Confirms staffing commitments and kickoff dates for each wave.
6. **Coupling_GPU_Specification.md** – Technical specification for the coupling framework and GPU execution model.
7. **FSI_Prototype_Plan.md** – Execution plan for the first coupling benchmark; check status annotations for progress.
8. **FSI_Field_Registration.md** – Detailed design for Step 1 of the FSI prototype (DualView/FieldRegistry wiring).
9. **AI_analysis_spec_comparison.md** – Background comparison of the two original AI analyses and specs; use as reference material.

## How to Use These Docs

- **For planners/architects:** Read in order 1 → 5 to align on architecture and resourcing, then 6 → 8 for feature-specific work.
- **For engineers picking up tasks:** Jump to the document associated with your wave and follow linked implementation notes.
- **For external agents (Claude/Gemini):** Start with item 1 to understand the macro design, then consult items 2–8 as you scope tasks. Reference item 9 for legacy vs. next-gen context when needed.

## Quick Links by Role

- **Architecture Steering Group:** Items 1–2, plus the latest updates in 5.
- **Wave Leads:** Items 3–5 for scheduling, 6–8 for detailed execution.
- **Coupling Taskforce:** Items 6–8, using item 9 to trace legacy behaviour.

Keep this index up to date when new documents are added or workflows change.
