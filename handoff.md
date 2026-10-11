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

## 2026-10-09 — Repository Rehabilitation and Live Acceptance

Current repo position:
- All nine rehabilitation tracks are implemented: bootstrap/lock, owned nodes,
  governance enforcement, network hardening, process lifecycle, CI/tests, model
  manifest, docs/assets, and repository administration.
- `./scripts/bootstrap.py --check --skip-deps`, `pip check`, and 8 repository
  tests pass. Pinned upstream result: 308 passed, 1 skipped, 1 deselected.
- Live managed start reported all services ready; ComfyUI reported MPS. A Jazzy
  workflow request without Adaptive credentials failed closed and ComfyUI
  history remained zero. Managed stop left Adaptive on port 8765.

Decisions and boundaries:
- Adaptive must accept `comfy.workflow.start` before ComfyUI receives a prompt.
- Terminal audit failure produces durable ignored local replay evidence.
- Unknown model source/licence values remain explicit and block redistribution
  claims; one interrupted 23.8 GB cache download is recorded in the model lock.
- A sphere is the intentional avatar fallback until a licensed GLTF is selected.

Immediate next actions:
- Commit and push the rehabilitation change set.
- With valid Adaptive credentials, capture a positive start/finish audit proof.
- Review model provenance metadata and the interrupted download.

Post-push verification:
- Commit `024b731` reached `origin/main` and the worktree was clean.
- GitHub created CI run `37980509621`, but started zero steps because the account
  is locked due to a billing issue. This is an external administration blocker,
  not a test failure. The runner image is pinned to Ubuntu 24.04 in the follow-up
  commit to avoid the announced `ubuntu-latest` migration.

## 2026-10-10 — Apple Silicon Baseline and Blender Reconnaissance

Current repo position:
- A governed Apple Silicon benchmark harness now covers Jazzy interaction,
  Branding Lab social, and Property Social quality workflows. On the M3 Ultra,
  PyTorch 2.5.1 MPS warm medians were 3.022 s, 25.667 s, and 72.967 s.
- Adaptive audit admission initially rejected the obsolete
  `execution.audit.v1` schema value. The emitter now follows the canonical
  `v1.1` contract and retains bounded structured rejection diagnostics.
- Adaptive topology is explicit: API on 8080 and streaming UI on 8765.
- The existing Jazzy GLB has all 52 ARKit morph targets, but no skin or animation;
  the visual mesh remains an early prototype rather than a production avatar.
- The existing `uvx blender-mcp` server is registered globally for Codex and
  will become available after a new Codex session.

Evidence and boundaries:
- `15 passed`; `pip check`, bootstrap check, and model manifest check pass.
- Baseline methodology and local report hashes are recorded in
  `docs/APPLE_SILICON_BASELINE.md`. Generated reports and images remain ignored.
- Comfy MCP's detached launcher currently uses comfy-cli's Python rather than
  this repo's `.venv` and fails on missing Pillow. The repo lifecycle manager is
  the verified launcher; fixing the MCP interpreter selection remains open.
- `/system_stats` is system-wide unified-memory evidence, not kernel-level MLX
  or zero-copy proof.

Immediate next actions:
- Build an isolated modern-PyTorch comparison environment without changing
  `.venv`, then repeat the three deterministic profiles.
- Use the Blender MCP in a fresh session to inspect and upgrade Jazzy's mesh,
  rig, idle animation, and viseme mapping.
- Profile FLUX Dev before selecting any operation for an MLX port.

Qwen production-baseline addendum:
- The remembered large model is `qwen_image_2512_bf16.safetensors`: 40.9 GB
  decimal (38.1 GiB), used with the 16.6 GB `qwen_2.5_vl_7b.safetensors`
  encoder and Qwen image VAE.
- The proven 1024x576/20-step route measured 113.729 s cold and 72.454 s warm
  median on PyTorch 2.5.1 MPS. All three outputs completed under governed
  Adaptive admission and passed visual inspection.
- A canonical 1328x1328/50-step maximum-quality probe took 1,208.675 s cold;
  the second queued run was deliberately interrupted after the workload
  mismatch was established. It is retained as bounded evidence, not promoted
  as the production benchmark.
- Qwen BF16 model, root text encoder, and VAE are now marked workflow-verified
  for MPS in the model lock. FP8 remains unsupported/operator-dependent.

## 2026-10-11 — Temporal MLX VAE Scaling

Current repo position:
- A reproducible temporal Wan/Qwen VAE harness now compares native Torch MPS
  and MLX using two governed 20-step, 1024x576 Qwen latent endpoints.
- The causal trials decoded 1, 5, and 13 RGB frames. MLX achieved 3.986x,
  3.442x, and 2.982x paired speedups with all fixed structure, per-frame,
  aggregate-quality, and temporal-difference gates passing.
- The predeclared 20 GiB memory gate failed: the thirteen-frame MLX path peaked
  at 36.89 GiB and the official allocator retained 82.53 GiB cached.
- An eager bounded-cache variant produced byte-identical MLX outputs, cleared
  retained allocator cache to zero, and remained 2.086x faster at thirteen
  frames. It did not reduce the transient 36.89 GiB peak.

Decisions and boundaries:
- This is decoder scaling evidence based on interpolated real latent endpoints,
  not evidence of video-generator motion quality or realtime production.
- Neither temporal mode is promoted because the original memory gate failed.
  Production remains on PyTorch 2.5.1 and the experiment stays in the isolated
  Torch 2.12/MLX environment.
- The next credible optimization is spatial tiling or activation-lifetime work
  inside the MLX decoder, retaining the same quality and memory comparator.

Evidence:
- `46 passed`; bootstrap and the 53-artifact model manifest pass.
- Official / bounded local report hashes:
  `0cf0074abc0d93f8e240635b0a024ce145b73fa37b52d6693d74cdf9d34f5531` /
  `833980aa5f4afaee8ae4b81fa701bd193af46f00bf853eb504bcc94203b7c9f5`.
- Changed files are the temporal benchmark and tests, capture-path correction,
  Qwen profile, runbook, and this append-only handoff entry.
