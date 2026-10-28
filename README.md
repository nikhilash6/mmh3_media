# MMH3 Media for ComfyUI

**Create video with audio. Continue a scene. Choose the best take.**

Custom nodes and ready-to-use workflows for MiniMax H3. Save `.mmh3` files to keep media, settings and generation state for later editing.

[Install](#installation) · [Start](#first-video) · [Workflows](#workflows) · [Models](#models-and-optional-features) · [Help](#quick-help)

## Installation

Use a ComfyUI installation with MiniMax H3 support. From its directory:

```console
cd custom_nodes
git clone https://github.com/einhorn13/mmh3_media.git ComfyUI_mmh3_media
```

Or download the ZIP and extract it into `ComfyUI/custom_nodes/ComfyUI_mmh3_media`, with `__init__.py` directly inside that folder. Keep one installed copy.

Add the required models, restart ComfyUI and refresh your browser. No extra dependencies are needed for the base package; keep ComfyUI's existing Python/PyTorch environment.

## First video

1. Drag [01 Generate](example_workflows/primary/01_generate.json) onto the canvas.
2. Select your diffusion model, text encoder and video/audio VAEs in **H3 Generate**.
3. Enter a prompt in **MMH3 Create**. Leave frame inputs empty for text-to-video, or connect first/last frames.
4. Set resolution and duration in **H3 Video Settings**. Start with a short clip.
5. Run. Save the video to share it and the `.mmh3` to continue working.

**Standard, 20 steps** is the starting preset. Turbo needs the matching acceleration LoRA. A selected task loads only its matching checkpoint.

## Workflows

| Workflow | Use it to |
| --- | --- |
| [01 Generate](example_workflows/primary/01_generate.json) | Generate from text, frames or references. |
| [02 Continue](example_workflows/primary/02_continue.json) | Extend a scene, reroll and review takes. |
| [03 Edit](example_workflows/primary/03_edit.json) | Edit video and audio independently. |
| [04 Refine](example_workflows/primary/04_refine.json) | Upscale the full frame or masked regions. |
| [05 Assemble](example_workflows/primary/05_assemble.json) | Join clips and export a sequence. |
| [06 Studio](example_workflows/primary/06_studio.json) | Build a long performance video scene by scene. |

In Studio: **Generate next → review → Accept take**, then Assemble when every scene is accepted. Project Manager keeps references, scene settings and saved takes together. Use continuation workflows for an uninterrupted shot.

**Keep source audio:** enable this checkbox in **H3 Generate** to protect a soundtrack during sampling and use it in the result, without mixing in model-generated audio. Connect the optional **source_audio** input, or use audio/video already attached to the packet. Edit and Refine keep source audio by default in the primary templates.

The pilot [F01 Ref2VA workflow](example_workflows/mmh3_f01_ref2va.json) also exposes **Keep source audio**, **Audio fit** and **Audio source** directly on **Generate H3 Video**, keeping its existing model, sampling, LoRA and optimization controls. Leave the source at `auto` for one soundtrack, or enter `audio:0` to protect the first audio reference (`audio_0`) when several are connected. Protection is off by default. The video and saved `.mmh3` receive the same selected track.

The [F02 Ref2VA continuation](example_workflows/mmh3_f02_ref2va_continuation.json) exposes **Audio mode** on **Generate H3 Video**:

- **Continue source** reads the next window of the full source saved by F01/F02. Load the last committed `.mmh3`; rerolling from the same parent reads the same window. The context belongs to the previous segment and is included only to guide generation.
- **New audio** places a new track at the first *new* frame, after that context. Connect a Load Audio output to **New audio**, or explicitly choose a reference in **New audio source** (`audio:0`, `video:0`, or resource ID). **New audio start (s)** skips a lead-in. **New audio fit** affects only this introduction: `cut` retains speed and keeps the full remaining source for later continuations; `stretch` consumes the whole new track within this segment's new frames.
- **Generate** retains the original F02 generated-sound behavior and does not maintain a protected source-audio clock.

Protected F02 clips and their parent carry absolute frame/sample ownership in `.mmh3`. In **F05 Video Stitch**, feed the committed packets in generation order and choose **Auto Seamless** or **Cut**. The stitcher recognizes this chain, removes each recorded context once and joins the retained PCM without another Audio VAE pass or duration-changing fades. Explicit **Crossfade** is blocked for these protected continuations. Other clip montages keep their existing transitions. Latent Stitch retains its latent-stream semantics; use Video Stitch to deliver the original waveform.

For one long song, use F01 **cut**, then F02 **Continue source**. Stretching the complete song into F01 has already consumed that song. At source EOF, further protected frames receive silence; audio is never looped or invented. Existing F01 archives can initialize the clock from their retained source; older F02 continuations without a clock require restarting from F01. Protected handover uses equal video/audio context and zero audio feather. Mouth timing is conditioned on the protected audio, but model lip-sync quality still needs a GPU review.

In [F05 Upscale + Stitch](example_workflows/mmh3_f05_latent_stitch_upscale.json), **Keep source audio** is on by default. Every HR refinement verifies and freezes the audio latent belonging to that same low-resolution part, preserving the exact HR context from the preceding part. Final delivery joins the input packets' original PCM using the F01/F02 clock and removes context once; it does not decode the final audio through Audio VAE or stretch the song again. Track replacements retain their recorded start frame. The report lists source windows and PCM ownership. Latent-only chains fall back to decoding their unchanged audio latents; partly missing PCM is an error unless source delivery is disabled. Refinement can still change mouth motion at strong denoise, so review visual lip-sync on the GPU even when the audio clock is preserved.

- **Audio fit · cut** trims excess audio or fills the missing tail with silence. **stretch** fits the complete track to the video length while preserving pitch; large tempo changes can introduce audible artifacts.
- **Audio source · auto** prefers the explicit audio input, then primary audio, primary video sound, active audio references, and finally sound embedded in active video references. Silent videos and disabled reference cards do not become automatic soundtrack candidates. A paired audio reference replaces its video's embedded soundtrack.
- Several candidate tracks require an explicit **Audio source**, such as `audio:0` or `video:0` from the reference cards, or a resource ID. Tracks are never mixed automatically. An explicit audio input overrides this selection.
- With protection enabled and no soundtrack, Generate produces silence. A missing explicit selection or unreadable/invalid soundtrack stops execution instead of silently falling back to generated sound.
- Delivery uses H3-compatible 32 kHz stereo; mono is duplicated to both channels. Generate and Refine retain the original source resource in the packet, so changing the duration can fit the original again. Protection does not guarantee lip-sync quality. FastH3 V2 does not support protected-audio generation; use a base H3 checkpoint and preset.

In **03 Edit**, **Keep source audio** controls the sampling mask; leave **Delivery audio** at `original_when_preserved` to retain the source waveform. With the checkbox off, advanced audio policies can regenerate all audio, intervals, or areas following the video mask. In **04 Refine**, the checkbox selects source delivery versus decoded latent audio; refinement itself always protects the source audio latent. Existing API prompts without the new fields retain their previous behavior.

**02 Continue** generates a new continuation soundtrack; it does not extend a supplied song beyond its recorded end. **05 Assemble** joins the supplied clip tracks, and crossfade intentionally blends their boundaries. **06 Studio** uses its master audio track for final assembly (converted to 32 kHz at project creation), with native audio timing guidance during generation. For a long video following one complete song, use Studio.

[Advanced examples](example_workflows) cover control, inpainting, latent stitching and clip bridges. [API templates](automation/workflows) support batch automation; `python automation_runner.py --help` lists runner options. Specialized paths may require additional nodes and models.

## Models and optional features

Choose the filenames actually installed on your computer.

| Component | ComfyUI folder |
| --- | --- |
| H3 FL2VA for text/frames, or Ref2VA for references | `models/diffusion_models/` |
| H3 text encoder | `models/text_encoders/` |
| H3 video VAE and audio VAE | `models/vae/` |
| Matching Turbo and creative LoRAs | `models/loras/` |

- **Load LoRAs:** search by part of a filename and build an ordered list. Custom replaces preset adapters; Extension adds to them. Auto uses the preset or recorded source adapters.
- **Continuation source LoRAs:** F02, F03, F04, controlled F04 and primary Continue connect the loaded archive to H3 Sampling's `source_packet`, and route its `clip` output to conditioning. Auto restores the recorded stack in order; Extension appends your list. Custom replaces source acceleration adapters and Disabled drops them, preserving creative LoRAs. Source opt-outs are respected; missing files, hash mismatches or unknown source stacks stop with an error. For an existing graph, add these connections or reload the updated example. Set Sampling's task family explicitly to match the conditioning family; its steps, sampler, scheduler and shifts still use the selected profile, not the source archive. FastH3 architecture recipes use their separate explicit refinement path.
- **Prompt resource labels:** Save/Load archive galleries show the native `<Picture 1>`, `<Audio 1>` and `<Video 1>` tags beside each active reference. Inactive references, context and results have no prompt tag. Tags reflect the archive's current active reference set; downstream reference edits can renumber them. These prompt tags differ from zero-based resource selectors such as `audio:0`.
- **Seed inheritance:** a seed override of `-1` keeps the archive's recorded seed, including `0`; a nonnegative override replaces it. The effective seed is used for noise and saved in the result archive. If the source has no seed, the resolver defaults to `0`.
- **FastH3 V2:** use a complete [Comfy checkpoint](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy) and the **FastH3 V2 - 8 steps** preset for text-to-audio-video without references. Native VSA is applied automatically; keep Load LoRAs in Auto. GPU quality and speed are not yet verified here.
- **Refine:** requires [H3 Latent Upscaler Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus) or the original [LBH upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) with `enable_temporal_chunking` / `force_unload`, and matching weights. The adapter detects the installed schema automatically. For LBH it disables internal temporal chunking and maps the unload setting to `force_unload`; Plus retains its own settings. Only video latents enter the learned upscaler.
- **Optimizations:** choose one attention strategy. FP16 accumulation and SageAttention can be enabled independently when their backends are available. Native Sol replaces the removed SolAttn_triton integration.

## Quick help

| Problem | Try this |
| --- | --- |
| Missing nodes or models | Check startup errors, model folders and selected filenames. |
| Out of memory | Reduce resolution or duration; generate long videos in segments. |
| Reopen a result | Use **MMH3 Load** and Refresh (R). Keep `path override` empty for the file list. |
| Preserve audio during generation/edits | Enable **Keep source audio**; in Edit keep delivery at `original_when_preserved`. Use **cut** or **stretch** to fit the duration. |
| Inputs changed after an update | Restart ComfyUI, refresh the browser and open the latest example JSON. |
| Old archive will not load | Schema-2 `.mmh3` archives are supported; pre-v0.3 files are not converted. |

To update a Git installation, run `git pull --ff-only` in this node folder. Keep personal workflows separately. For ZIP installs, replace the package with the new version.

For a bug report, include the ComfyUI version, workflow JSON and first console error.

Licensed under [LICENSE](LICENSE).
