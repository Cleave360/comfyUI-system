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

## 2026-10-11 — MLX Decoder Activation-Lifetime Follow-up

Current repo position:
- Layer-boundary, residual-operation, and final-stage spatial-tail candidates
  were evaluated against the same governed temporal endpoints and fixed gates.
- All three preserved byte-identical MLX output and zero retained allocator
  cache. Thirteen-frame peak memory fell from 36.89 GiB to 30.45, 25.05, and
  24.74 GiB respectively; the last remained 2.005x faster than Torch MPS.
- The fixed 20 GiB gate still failed. Five-frame spatial tiling reached
  20.14 GiB and thirteen frames reached 24.74 GiB.

Decisions and boundaries:
- Final-stage-only spatial tiling is rejected as a sufficient memory solution.
- No production promotion is authorized. The strongest retained diagnostic
  candidate is operation-boundary materialization, with spatial tiling kept for
  exact comparative evidence.
- Next work should attribute peak memory across earlier upsampling stages and
  causal feature caches before attempting a broader tile or cache redesign.

Evidence:
- Layer / operation / spatial report hashes:
  `875ec451cfa1a35bba15932ec1f30f1df5ca02fb3d4132ccc856c1a1831e2062`,
  `3f7c2db5b0d1e7ae79dbcab572f4c7fa491aa313d687763b23d51d241880106b`,
  and `effffbd08d457bccda2aadbaebbd8f9f7df47e6b065bdc5a03fa8d68520aefd1`.

## 2026-10-11 — MLX Temporal Memory Gate Passed

Current repo position:
- Per-operation telemetry isolated the remaining peak to old/new causal-cache
  overlap followed by the four-slice final spatial resample.
- Consuming each cache entry after its single use reduced thirteen-frame peak
  from 25.045 to 20.579 GiB. Chunking only the non-causal `upsample2d` across
  temporal slices reduced it to 19.908 GiB.
- Two independent full sweeps passed every original quality, speed, structure,
  and 20 GiB memory gate. Thirteen-frame speedups were 2.101x and 2.105x, with
  byte-identical official-MLX outputs and zero retained allocator cache.

Decisions and boundaries:
- Promote the cache-consuming/chunked-resample path only to a reproducible
  component candidate. It is not yet a production or video-quality result.
- Do not chunk the two causal 3D resamplers; only the final independent spatial
  resample is split across temporal slices.
- Next acceptance requires native latents from a real video model and a
  repeated isolated-server endurance run.

Evidence:
- Attribution / decisive / repeat report hashes:
  `987841b69a5a719dc6753b0bcaec4906c275ea69ae89cdd26381c8248da7e02e`,
  `63846ae6588e46276e98d7fae89e2cd29c93f6ea790f3033626b8ce66ed2e799`,
  and `8012335a019936d7889d3e0d78cdf7874cdae0037e5dbcf3bdbe8a52b354bb0d`.

## 2026-10-11 — Native Wan Video and Newer-Torch UniPC Acceptance

Current repo position:
- A minimal governed Wan VACE 1.3B workflow now produces 17 frames at 832x480
  in four steps and captures the genuine `[1,16,5,60,104]` temporal latent.
- Production PyTorch 2.5.1 failed inside unchanged UniPC because MPS does not
  implement `_linalg_solve_ex`. The isolated PyTorch 2.12.1 runtime passed a
  direct solve control and completed the same governed UniPC graph in 56.209 s.
- The real Wan latent and weights passed the MLX decoder gates twice. MLX
  medians were 4.232/4.230 s versus Torch 10.301/11.274 s, for 2.434x/2.666x.
  Both repeats had identical outputs, 13.596 GiB peak, and zero retained cache.

Decisions and boundaries:
- Do not fork or rewrite ComfyUI UniPC; it already calls the correct public
  Torch API. Treat newer PyTorch as the candidate fix.
- Do not replace production `.venv` yet. The isolated environment blocks the
  incompatible production torchaudio binary and has not passed the whole stack.
- Native temporal component acceptance is complete. End-to-end MLX server
  integration and repeated full-workflow endurance remain open.

Evidence:
- `12 passed` focused tests before the measured run; final repository validation
  was `50 passed`, bootstrap PASS, pip check clean, and 53 model artifacts PASS.
- Capture / governed workflow / decisive / repeat SHA-256:
  `68feda22d97e4e37c2d09d49a489053e82e4317829da7b9abd0aa703cf415f62`,
  `23739633c8107cc16fb89492c3e913bd3ae966e3c736ea8397cb86d147a1021b`,
  `2701d0d54bab3d35f58f893af17196ef9fc0bb5af30c0a30e724efc737f69c3a`,
  and `abfd9133e030feb0e37711decc92e4d98d31c102d8288d669dcdd0d5f121a312`.

## 2026-10-11 — Alternating Full-Stack Wan MLX Acceptance

Current repo position:
- An isolated Torch 2.12 ComfyUI server now alternates the stock Wan decoder
  and the cache-consuming/chunked-resample MLX candidate in a fixed ten-run
  sequence. The sampler, conditioning, model, prompt, and save path are shared.
- Warm-ups and a stock/stock control precede three same-seed pairs with order
  alternating. A cache-bust node forces one complete temporal decode per run.
- All fixed gates passed: exact control; matching paired latent hashes; 17 valid
  frames per run; global/per-frame/temporal quality; mode accounting; 3/3
  end-to-end wins; 4.588x paired median VAE speedup; 13.596 GiB peak; zero MLX
  cache retention.

Decisions and boundaries:
- This is full-stack directional evidence because MLX reduced all three paired
  workflow times. It is not a stable production throughput number.
- Shared-system timing drift was severe. The least-contended pair saved 4.011
  seconds (42.134 to 38.123); later stock runs slowed far beyond baseline while
  free unified memory remained above 353 GiB. Do not quote the 72.273-second
  median saving as normal performance.
- Production remains on Torch 2.5.1. The candidate still lacks compatible audio
  dependencies and needs runtime-drift diagnosis before promotion.

Evidence:
- Full-stack benchmark / server ledger SHA-256:
  `efb896141c0d73f6d6bc62bce2b9fe597021b2e6be07a23fd4d1069209ca4a8d` /
  `8ac767e6dd11ece6cccb636672277db3a9f641d1073dcdc0a9e73f76c16d4bdb`.
- Manual comparison of all three midpoint pairs found no visible stock/MLX
  divergence; first/middle/last inspection confirmed temporal motion.

## 2026-10-11 — Wan Sustained-Load Diagnosis and Passing Candidate

Current repo position:
- Decoder-only endurance passed with stable 10.324-second Torch and
  4.215-second MLX medians (2.449x), ruling out intrinsic VAE drift.
- Repeated stock full workflows at two-second spacing reproduced the collapse:
  sampler p95/median 1.502, total p95/median 1.542, and 6.572 GiB Torch driver
  growth. Fixed-seed output hashes remained identical.
- `torch.mps.empty_cache()` eliminated stock driver growth but did not stabilize
  timing. A 30-second gap made timings almost perfectly flat despite 29.977 GiB
  reported driver growth. Coarse macOS telemetry reported no thermal or memory
  pressure warning, so the evidence supports sustained-load recovery behavior
  without proving a thermal cause.
- All-MLX decode at two-second spacing stabilized timing. Combining it with the
  post-decode Torch cache boundary passed every fixed gate: 24.129-second total
  median, 1.165 p95/median, 1.083 last-half/first-half ratio, and 56 MiB driver
  growth across six measured workflows.

Decisions and boundaries:
- Retain MLX decode plus the cache boundary as the strongest isolated candidate.
  It is not a production promotion: PyTorch 2.12 audio compatibility and a
  longer soak across image/video workflows remain open.
- Do not interpret `torch_driver_allocated_memory` as a direct throttling cause;
  the cache-clear and cooldown interventions falsified that simple model.
- Production services and the Torch 2.5.1 environment remain unchanged.

Evidence:
- Decoder-only / stock / stock-cache / cooldown / MLX / passing-combined report
  hashes: `803a6cc0dbe161ffc51a8f5f7f7a483db8274442a0248fc646401d1da18e1521`,
  `805258e6868b2bad66be4552b01d9204baf735c0d5c7ad12a03517b0fe553c95`,
  `061861b153477196b63eeeaccd4ca43a4cbf717ff11ce0b73d7074c55adfe6ee`,
  `b46437652284dd7a27d521bc9737e2a8f70f231798ac2d8b38c5d71a6dfe172a`,
  `22f204aaea64e4348f292574687d161d8a95892bdd9ab9923a0076f204c82568`,
  and `7e71fc639946233029ac931b9b4a2f62e5b0395f53ea43f970c153f7a9667c93`.

## 2026-10-11 — Mixed Qwen/Wan Soak and Audio Restoration

Current repo position:
- The isolated Torch 2.12.1 requirements now pin stable-ABI TorchAudio 2.11.0.
  Direct resampling passed, ComfyUI loaded its audio extras without import
  failures, and a governed `EmptyAudio -> SaveAudio` workflow succeeded.
- One process completed Qwen/Wan warm-ups followed by six strictly alternating
  measured pairs. Every workflow was valid, all six Qwen pixel hashes matched,
  and all six 17-frame Wan pixel hashes matched.
- Memory gates passed: zero Torch driver growth, zero retained MLX cache,
  13.596 GiB MLX peak, and no process-RSS growth.
- Timing p95 gates failed. Qwen total p95/median was 1.709 and Wan was 1.492;
  sampler ratios were 1.881 and 1.671. The large excursion was concentrated in
  pair two, after which both workloads recovered. Last-half/first-half ratios
  were 0.974 for Qwen and 0.928 for Wan, so monotonic-drift gates passed.

Decisions and boundaries:
- Audio compatibility is restored for the isolated runtime, but mixed-workload
  timing stability is not accepted. Do not promote it to production.
- Treat the result as an oscillatory sustained-load/model-switching problem,
  not a correctness or allocator-retention failure.
- The next discriminator should compare process isolation or a recovery-aware
  queue policy against this exact mixed sequence.

Evidence:
- Mixed soak / server / isolated requirements SHA-256:
  `9bf5981684289ace594664bba46cdbb915083491045ffd8e888b160700ffc277`,
  `af2100b26ca12f704cd56c1f00635cc49f7303893c8f8dca6245ec6e90124702`,
  and `fd9831937967576a6b8c90145f604efec73dbf475dfe99ff4c2c6f1bee4b138b`.

## 2026-10-11 — Persistent Qwen/Wan Process-Isolation Experiment

Current repo position:
- A new governed harness runs the same mixed sequence against two persistent
  isolated ComfyUI workers, with independent PIDs and per-worker ledgers.
- The Qwen worker now optionally records non-Wan VAE decode timing/memory and
  clears reclaimable MPS cache after every image decode. The Wan worker retains
  native MLX VAE decode plus its existing post-decode cache boundary.
- The complete two-warm-up/six-pair run preserved valid deterministic image and
  video output, stable late-run medians, separate PIDs, audio availability,
  bounded combined RSS, and zero retained MLX cache.
- Process isolation failed the promotion gates. Qwen total/sampler p95 ratios
  were 2.373/2.418, Wan total was 1.401, Qwen total median speedup was only
  1.180x, and the Qwen worker accumulated 4.141 GiB Torch driver allocation.
  The second Qwen measurement reached 318.972 seconds, after which totals
  stabilized near 134 seconds.

Decisions and boundaries:
- Reject persistent process isolation as a sufficient performance fix. It
  removes model-eviction ambiguity and improves Wan total latency, but does not
  isolate the shared Metal device's sustained-load excursion.
- Preserve this as negative evidence; do not promote either experimental
  worker or the new cache setting into production.
- The next credible discriminator is a recovery-aware queue/cooldown policy
  keyed to the shared device, not another model-residency variant.
- Production services remain unchanged on 8188, 8075, and 8070.

Evidence:
- Isolated soak / Qwen worker / Wan worker SHA-256:
  `2218e083201ddcd5b8927b7a003c1619171e4a7fc0dc8a9c334f80375d6b7862`,
  `0db362536f0e8139b0e4e491f3ad7699dd0afce899a2d9c896c97105b37905ad`,
  and `4875de150561795a0927f77b7438c2183cef01de80f8daeeeb8a7042a72c4b3b`.

## 2026-10-12 — MPS Recovery Policy Screen and Metrics GUI

Current repo position:
- A read-only loopback GPU metrics dashboard now visualizes per-run workflow
  duration, MPS probe latency, unified-memory availability, recovery waits,
  outputs/hour, and Apple public thermal-pressure state. Historical reports
  correctly show thermal state as unavailable because it was not captured.
- A governed recovery harness ran four fresh-server arms with two warm-ups and
  four measured Qwen/Wan pairs each: fixed 2 seconds, adaptive, fixed 30
  seconds, and fixed 10 seconds. All 40 workflows were valid and deterministic.
- Adaptive passed every predeclared gate: 30.582 outputs/hour, 1.033 Qwen and
  1.001 Wan p95/median, 100% readiness, and 1.128x fixed-2s throughput.
- Fixed 10 seconds narrowly ranked first at 30.763 outputs/hour. Fixed 30
  seconds had the lowest and flattest raw render times but only 29.348
  outputs/hour after waiting. Fixed 2 seconds failed Qwen stability at 1.443.
- A real post-run MPS probe verified `ProcessInfo.thermalState=nominal`, Low
  Power Mode false, and explicit null Celsius/throttle-temperature fields.

Decisions and boundaries:
- Preserve adaptive as a positive candidate, not a production promotion.
  Four pairs and fixed policy order are screening evidence only.
- The next confirmation should counterbalance adaptive and fixed 10 seconds.
  Their 0.59% throughput difference is too small to call a winner here.
- Treat `serious` as Apple's public performance-impact boundary and `critical`
  as significant impact. Do not invent a GPU temperature or throttle cutoff;
  the public signal is system-wide thermal pressure, not a GPU thermometer.
- Production services and startup paths remain unchanged.

Evidence:
- Adaptive / fixed-2 / fixed-10 / fixed-30 SHA-256:
  `26839703dc8485d331d8f58eaea2c11f8c670720e7e838494fa6b9568aa2fffb`,
  `82b96d0cbbc1ffce971f2927b2f4454cbde795c64d50a54ddf3b96bfeb78dddc`,
  `781fb4140470b9f234f4e46e80af10069046291a8c68fddcdc0da9f964e7030e`,
  and `e9a3881eaeaa71ee4bb9158e093331f9810b5cd4fa2601806c922d69a95589b0`.
- Comparison / matrix manifest SHA-256:
  `b28d853b9e7c0f103c32609b599a468ea8a04aab9c7d5e2bf785c8ec18b41379`
  and `7770330cc6cfe08da9882203b142be6b14205c5918df43fef2367a303a8e596a`.
