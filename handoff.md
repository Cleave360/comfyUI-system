# Handoff Protocol

**Collaboration Between System Entities**:

1. **Gemini Co-Architect (Design & Monitoring)**: Uses MCP parameters to read graphs, pull models, and monitor `psutil` memory pressure. Determines workflow architecture.
2. **Copilot Engineer (Low-Level Systems)**: Handles environment conflicts, driver alignments, script creation, and dependency resolution.
3. **User (Director)**: Approves implementations, specifies visual logic, guides testing parameters.
4. **Kindred iPhone Core**: Awaiting stable connection endpoints to fetch generation payloads and workflow states once Swarm Integration unlocks.

## 2026-05-09 — Adaptive KI Alignment (Cross-Repo Enforcement)

Canonical source docs (must be read before implementation):
- /Users/geofflundholm/Documents/adaptive_layer/docs/antigravity_knowledge_instruction_v1.md
- /Users/geofflundholm/Documents/adaptive_layer/docs/kindred_adaptive_contract_v1.md
- /Users/geofflundholm/Documents/adaptive_layer/docs/execution_context_envelope.md
- /Users/geofflundholm/Documents/adaptive_layer/docs/production_sprint_14_day_checklist.md

Cross-repo non-negotiables:
1. Governed calls must include full envelope: tenant/workspace/project/run/principal/ui.
2. Contract status taxonomy is strict: success | error | rejected.
3. Pause semantics: event_type=*.pause, policy_decision=halt, status=rejected, payload.phase=await_*.
4. No status=paused in Adaptive append payloads.
5. Unknown/missing tenant in governed mode must fail closed.
6. No contract drift without updating canonical docs and smoke/proof scripts.

Evidence expectation on each meaningful change:
- Command/test run used
- Result artifact path
- Streams/ledger proof notes
- Exact files changed

Repo-specific enforcement (ComfyUI):
- Node/workflow actions should map to contract-safe event tuples before audit append.
- Use lifecycle phase fields (await_*, in_progress, completed) while preserving strict status values.
- Avoid direct non-governed execution for controlled production paths.

## 2026-07-18 — Copilot Session Start (Branding + Runtime Stability)

Introduction:
- I am GitHub Copilot (GPT-5.3-Codex), operating as Copilot Engineer for low-level systems, integration reliability, and workflow execution hardening.

Current repo position:
- Working in the ComfyUI workspace with active focus on ComfyUI runtime startup/reliability, Jazzy voice/image routing, and workflow readiness for image generation.
- Recent state confirms ComfyUI startup is scripted on port 8188, Jazzy routing has been moved to port 8075, and brand workflow expansion has been added for Kindred campaign modes.

Shared vision:
- Build a stable, governed production creative stack where voice and prompt intent reliably trigger the right workflow, outputs stay brand-consistent, and all controlled paths remain contract-safe and audit-ready.

## Coordination Note — Multi-Repo Vision Sync

Context:
- Branding Lab is the orchestration hub across studio repos; this ComfyUI repo is a spoke focused on generation runtime, workflow execution, and production handoff reliability.
- Cross-repo production tracks include Property Social media outputs, neural-chronicles.uk videos, and digital monthly magazine pipelines coordinated from the hub.

Use this short structure for each agent/session entry:
- Introduction: role and operating focus.
- Current repo position: what is active now, what is stable, what is blocked.
- Expanded vision: 30-60 day direction across ComfyUI + Branding Lab + Property Social media.
- Immediate next actions: top 3 implementation items with owner.
- Evidence: commands/tests run, artifact paths, and exact files changed.

Suggested headings to copy:
- Introduction
- Current Repo Position
- Expanded Vision
- Immediate Next Actions
- Evidence
