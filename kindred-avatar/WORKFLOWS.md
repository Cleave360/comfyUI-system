# Jazzy Image & Motion Workflows

Curated presets Jazzy can call on demand. Each preset has a matching `[IMAGE:<mode>: prompt]` syntax so the backend instantly knows which ComfyUI graph to queue.

| Mode | File | Steps / Time | Resolution | Stack | Ideal Use | Syntax |
|------|------|--------------|------------|-------|-----------|--------|
| Quick Sketch | `flux_quick.json` | 2 steps / ~15s | 512×512 | FLUX Schnell | Thumbnail ideation, scribbles | `[IMAGE:quick: …]` |
| Standard | `flux_standard.json` | 4 steps / ~30s | 1024×1024 | FLUX Schnell | General purpose default | `[IMAGE:standard: …]` or `[IMAGE: …]` |
| High Quality | `flux_quality.json` | 20 steps / ~2 min | 1024×1024 | FLUX Dev | Final polish, hero art | `[IMAGE:quality: …]` |
| Portrait Studio | `flux_portrait.json` | 25 steps / ~2 min | 832×1216 | FLUX Dev + portrait prompt boost | Character close-ups | `[IMAGE:portrait: …]` |
| Cinematic Landscape | `flux_landscape.json` | 20 steps / ~90s | 1344×768 | FLUX Dev + wide prompt boost | Environments, vistas | `[IMAGE:landscape: …]` |
| Product Hero | `flux_product.json` | 16 steps / ~45s | 1152×864 | FLUX Dev + studio lighting prompt | Pack shots, hardware, apparel | `[IMAGE:product: …]` |

## How to Prompt Jazzy

1. **Name the workflow** so Jazzy can switch modes: “Jazzy, use the product hero workflow for a wearable AI pin on marble.”
2. **Add required details**: subject, palette, era, camera, lighting, mood. Jazzy mirrors those details inside the `[IMAGE:<mode>: …]` tag.
3. **Give guardrails** if needed: “Keep it monochrome” or “use depth of field.” Jazzy adds those to the prompt block.
4. **Stack follow-ups**: Jazzy tracks the last workflow per conversation; saying “same workflow, but neon blue” keeps parameters while swapping text.

Example utterances:
- “Generate a quick sketch of a modular studio set.” → `[IMAGE:quick: modular studio set, loose pencil lines, high contrast]`
- “I need a cinematic landscape of Mars dunes at dusk.” → `[IMAGE:landscape: sweeping Martian dunes at dusk, cinematic lighting, volumetric fog]`
- “Use the product hero workflow for a translucent AR headset on gloss black.” → `[IMAGE:product: translucent AR headset, gloss black plinth, soft rim light, reflections controlled]`

## Workflow Notes

### Quick Sketch (`flux_quick.json`)
- **What changes**: Only 2 steps, 512² latent for instant feedback.
- **Prompt tip**: Emphasize medium (“marker sketch”, “blueprint”).
- **When to ask Jazzy**: “Riff 3 variations quickly…”

### Standard (`flux_standard.json`)
- **Balanced**: 4 steps Schnell = best mix of time/detail.
- **Prompt tip**: Keep under 40 tokens; Jazzy can append descriptors if you say “feel free to polish the prompt.”

### High Quality (`flux_quality.json`)
- **Heavier**: FLUX Dev, 20 steps, outputs to `jazzy_hq_*`.
- **Prompt tip**: Provide composition + finish e.g. “35mm photo, Leica lens, natural grain.”
- **Usage cue**: “Jazzy, give me a high-quality render of …”

### Portrait Studio (`flux_portrait.json`)
- **Extras**: Auto text appends `professional portrait, detailed facial features, studio lighting`.
- **Aspect**: 832×1216 vertical.
- **Prompt tip**: Mention age, styling, camera height.

### Cinematic Landscape (`flux_landscape.json`)
- **Extras**: Adds `wide angle, cinematic composition, detailed environment`.
- **Aspect**: 1344×768 anamorphic.
- **Prompt tip**: Anchor time of day + atmosphere.

### Product Hero (`flux_product.json`)
- **New**: 16-step FLUX Dev with `premium product hero shot, seamless backdrop, softbox rim lighting, hyper crisp surfaces` baked into the text encoder.
- **Aspect**: 1152×864 for presentation decks.
- **Prompting Jazzy**: “Run the product hero workflow for a smartwatch with bronze trim, floating above a ripple of water.”
- **Output**: Saved as `jazzy_product_*` so investors can spot hero renders instantly.

## Motion Loop (AnimateDiff Spec)

Motion lives in ComfyUI via AnimateDiff Evolved. The JSON export will land as `animatediff_motion_loop.json` once the node graph is finalized; until then use these build notes:

1. **Dependencies**
   - Extension: `Kosinkadink/ComfyUI-AnimateDiff-Evolved`
   - Base checkpoint: SD 1.5 or SDXL finetune that matches your brand.
   - AnimateDiff motion module (e.g., `mm_sd_v15.ckpt`).
   - Optional Motion LoRA (pan, zoom, tilt) dropped into `models/animatediff_motion_lora/`.
2. **Graph outline**
   - `CheckpointLoaderSimple` → `CLIPTextEncode` (prompt + negative).
   - `ADE_AnimateDiffLoaderV1` + `ADE_AnimateDiffSettings` (frame count, fps=12, loop blend on).
   - `AnimateDiffSampler` with 16–24 frames.
   - `VAEDecode` → `ImageSequenceSave` (GIF/MP4).
3. **Prompting Jazzy**
   - Until the JSON is wired into the backend, have Jazzy craft the full AnimateDiff prompt and shot list: “Jazzy, help me storyboard a 12-frame looping animation of the Jazzy avatar smiling—list keyframes and camera notes.”
   - Once the workflow is exported, we’ll enable `[MOTION:loop|prompt|frames]` syntax so Jazzy can trigger it exactly like the still workflows.

## Files & Maintenance

Located in `backend/workflows/`:
- `flux_quick.json`
- `flux_standard.json`
- `flux_quality.json`
- `flux_portrait.json`
- `flux_landscape.json`
- `flux_product.json`
- _Coming soon_: `animatediff_motion_loop.json`

To add more:
1. Drop the new JSON into `backend/workflows/`.
2. Register it in `server_voice.py → self.available_workflows`.
3. Mention it inside the system prompt and this document.

## Performance & GPU Notes

- Schnell presets sit under ~12 GB VRAM; Dev presets peak around 18 GB.
- Run one workflow at a time; ComfyUI queues everything else.
- Keep `noise_seed` randomization on (`server_voice` already passes one) so investor demos feel fresh.

## Next Up

- [ ] Export the AnimateDiff loop graph and wire `[MOTION:…]` handling.
- [ ] Add ControlNet-assisted variants (pose/depth) for live mocap tie-ins.
- [ ] Slot in upscaler + face-detail passes for the HQ and Portrait presets.
