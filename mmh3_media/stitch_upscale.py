from __future__ import annotations

import json
import math
import torch
from .errors import MMH3ResourceError
from .h3 import make_nested_tensor, nested_parts, validate_h3_av_latent
from .latent_stitch import inspect_h3_latent_stitch_packets
from .latent_upscale import build_latent_upscale_refine_sampling, _source_upscale_identity, prepare_packet_latent_upscale
from .lora_provenance import get_generation_loras
from .upscale_overrides import parse_upscale_sigmas
from .upscaler_adapter import UPSCALER_NODE, build_upscaler_inputs
from .audio_timeline import timeline_stitch_windows
from .audio_preservation import checked_audio
from .stitch import pcm_boundary, _normalize_audio_length

DEFAULT_UPSCALER = 'minimax_h3_latent_upscaler_3d_bf16.safetensors'
UPSCALE_ASPECTS = ('Source', '16:9', '9:16', '1:1', '4:3', '3:4', '2:3', '3:2')


def source_audio_windows(packets, keep_source_audio=True):
    """Resolve PCM ownership before allocating HR models or sampler graphs."""
    compatibility = inspect_h3_latent_stitch_packets(packets)
    if not compatibility.ready:
        raise MMH3ResourceError('Upscale audio: ' + '; '.join(compatibility.reasons))
    if not keep_source_audio:
        return None
    available = [p.get_primary('audio') is not None for p in packets]
    if not any(available):
        return None  # Latent-only archives retain the original VAE decode path.
    if not all(available):
        raise MMH3ResourceError('Some upscale segments have no source PCM. Restore their audio or disable Keep source audio to decode the stitched audio latent.')
    windows = timeline_stitch_windows(packets)
    if windows is None:
        windows = []
        end = 0
        for index, fact in enumerate(compatibility.facts):
            trim = int(fact['video_handover_frames']) if index else 0
            start = end - trim if index else int(fact.get('absolute_start_frame') or 0)
            end = start + int(fact['frames'])
            windows.append({'trim_frames': trim, 'trim_samples': pcm_boundary(start + trim) - pcm_boundary(start),
                            'raw_start_frame': start, 'end_frame': end})
    for packet, window in zip(packets, windows):
        if window['end_frame'] <= window['raw_start_frame'] + window['trim_frames']:
            raise MMH3ResourceError('Upscale audio context consumes the complete segment')
        audio = packet.ref(packet.get_primary('audio')['id']).materialize()
        wave, rate = checked_audio(audio)
        if rate != 32000 or wave.shape[1] != 2:
            raise MMH3ResourceError('Upscale source PCM must be 32 kHz stereo; normalize the source packet first')
        expected = pcm_boundary(window['end_frame']) - pcm_boundary(window['raw_start_frame'])
        if abs(wave.shape[-1] - expected) > math.ceil(32000 / 24):
            raise MMH3ResourceError('Upscale source PCM duration differs from its recorded video window')
    return windows


def assemble_upscale_source_audio(packets, stitched_latent, assembly_report):
    """Copy saved PCM windows once; HR audio latents never become delivery PCM."""
    windows = source_audio_windows(packets)
    if windows is None:
        raise MMH3ResourceError('Source PCM disappeared after upscale preparation')
    info = validate_h3_av_latent(stitched_latent, strict_audio_length=True)
    start = windows[0]['raw_start_frame']
    end = windows[-1]['end_frame']
    if info.frames != end - start:
        raise MMH3ResourceError('Stitched video duration differs from the source audio timeline')
    from .pcm_storage import allocate_pcm, copy_pcm
    output = allocate_pcm(2, pcm_boundary(end) - pcm_boundary(start))
    cursor, pieces = 0, []
    for index, (packet, window) in enumerate(zip(packets, windows)):
        resource = packet.get_primary('audio')
        audio = packet.ref(resource['id']).materialize()
        wave, rate = checked_audio(audio)
        if rate != 32000 or wave.shape[1] != 2:
            raise MMH3ResourceError('Upscale source PCM must be 32 kHz stereo; normalize the source packet first')
        raw_samples = pcm_boundary(window['end_frame']) - pcm_boundary(window['raw_start_frame'])
        wave, adjustment, delta = _normalize_audio_length(wave, raw_samples, tolerance_samples=math.ceil(32000 / 24))
        skip = window['trim_samples']
        piece = wave[..., skip:]
        copy_pcm(output, cursor, piece)
        pieces.append({**window, 'index': index, 'source_packet_id': packet.manifest['id'],
                       'audio_resource_id': resource['id'], 'audio_revision': resource['content']['revision'],
                       'source_sample_start': skip, 'source_sample_end': raw_samples,
                       'output_sample_start': cursor, 'output_sample_end': cursor + piece.shape[-1],
                       'audio_adjustment': adjustment, 'audio_adjustment_samples': delta})
        cursor += piece.shape[-1]
    if cursor != output.shape[-1]:
        raise MMH3ResourceError('Upscale PCM ownership windows contain a gap or overlap')
    report = {**assembly_report, 'audio_delivery': {
        'policy': 'source_pcm', 'sample_rate': 32000, 'channels': 2, 'samples': cursor,
        'source_start_frame': start, 'source_end_frame': end,
        'context_policy': 'remove_once_at_recorded_boundary', 'audio_vae_roundtrip': False,
        'windows': pieces}}
    return {'waveform': output, 'sample_rate': 32000}, report


def resolve_stitch_upscale_geometry(packets, resolution):
    """Read source geometry from the primary MMH3 latent, not editable metadata."""
    report = inspect_h3_latent_stitch_packets(packets)
    if not report.ready:
        raise MMH3ResourceError('Upscale + Stitch: ' + '; '.join(report.reasons))
    mode = resolution.get('resolution', 'Scale')
    if mode == 'Scale':
        options = dict(geometry_mode='scale', scale=float(resolution.get('scale', 2.0)))
    elif mode == 'Megapixels':
        mp = float(resolution.get('megapixels', 1.0))
        aspect = resolution.get('aspect_ratio', 'Source')
        if not math.isfinite(mp) or not 0.1 <= mp <= 8.0:
            raise MMH3ResourceError('Target megapixels must be within 0.1–8.0')
        if aspect not in UPSCALE_ASPECTS:
            raise MMH3ResourceError(f'Unknown aspect ratio: {aspect}')
        if aspect == 'Source':
            options = dict(geometry_mode='target_megapixels', target_megapixels=mp)
        else:
            x, y = map(int, aspect.split(':'))
            area = mp * 1024 * 1024
            options = dict(geometry_mode='target_dimensions',
                target_width=max(32, round(math.sqrt(area * x / y) / 32) * 32),
                target_height=max(32, round(math.sqrt(area * y / x) / 32) * 32))
    elif mode == 'Dimensions':
        options = dict(geometry_mode='target_dimensions',
            target_width=int(resolution.get('width', 1344)),
            target_height=int(resolution.get('height', 768)))
    else:
        raise MMH3ResourceError(f'Unknown resolution mode: {mode}')
    plan = prepare_packet_latent_upscale(packets[0], **options).plan
    if max(plan.target_width, plan.target_height) > 4096:
        raise MMH3ResourceError('Upscaler supports at most 4096 pixels per axis')
    return plan


def restore_upscale_av(sampled, target):
    """Restore protected video and exact target audio; discard transient sampler masks."""
    actual = validate_h3_av_latent(sampled, strict_audio_length=True)
    expected = validate_h3_av_latent(target, strict_audio_length=True)
    if actual.video_shape != expected.video_shape or actual.audio_shape != expected.audio_shape:
        raise MMH3ResourceError('HR sampler changed the segment AV geometry')
    video, _ = nested_parts(sampled['samples'])
    original, audio = nested_parts(target['samples'])
    if 'noise_mask' not in target:
        raise MMH3ResourceError('HR restore requires the protected upscale target')
    mask, _ = nested_parts(target['noise_mask'])
    video = torch.where(mask.to(device=video.device) == 0,
                        original.to(device=video.device, dtype=video.dtype), video)
    return {'samples': make_nested_tensor([video, audio])}


def validate_sources(packets, width, height, denoise):
    report = inspect_h3_latent_stitch_packets(packets)
    if not report.ready:
        raise MMH3ResourceError('Upscale + Stitch: ' + '; '.join(report.reasons))
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise MMH3ResourceError('Target width/height must be positive multiples of 32')
    recipes = []
    for index, (packet, fact) in enumerate(zip(packets, report.facts), 1):
        _source_upscale_identity(packet)
        if width < fact['canvas'][0] or height < fact['canvas'][1]:
            raise MMH3ResourceError(f'Segment {index}: target must not shrink either source dimension')
        if get_generation_loras(packet) is None:
            raise MMH3ResourceError(f'Segment {index}: source LoRA provenance is unknown; record the actual list first')
        recipes.append(build_latent_upscale_refine_sampling(packet, denoise_override=denoise))
    frames = sum(int(f['frames']) - int(f['video_handover_frames']) for f in report.facts)
    return recipes, f'{len(packets)} segments · {width}×{height} · {frames} frames / {frames / 24:.2f}s'


def build_stitch_upscale_expansion(packets, *, width, height, denoise, models, clip,
                                   video_vae, audio_vae, upscaler_model=DEFAULT_UPSCALER,
                                   graph_builder_factory=None, steps_override=0, manual_sigmas='',
                                   attention='Default', fp16_accumulation='Default', force_unload=True,
                                   upscaler_api='plus_v1', decode_mode='vae', trt_decoder='auto', sage_attention='disabled', sage_allow_compile=False, turbo_loras_json='', streaming='auto', keep_source_audio=True):
    if not 0 <= steps_override <= 100:
        raise MMH3ResourceError('Upscale steps must be within 0–100 (0 inherits source)')
    explicit_sigmas = parse_upscale_sigmas(manual_sigmas)
    selected_attention = dict(attention) if isinstance(attention, dict) else {'attention': attention}
    attention_label = str(selected_attention.get('attention') or 'Default')
    is_vdn = attention_label.startswith('VDN-H3') or attention_label == 'vdn_h3'
    if is_vdn:
        raise MMH3ResourceError(
            'VDN-H3 is not enabled for Upscale + Stitch low-sigma refine: this path truncates the '
            'scheduler with denoise<1, so it cannot guarantee the trained 8-NFE DMD or ~50-NFE Stage-B '
            'trajectory. Use a stock/SLA/Sol refine path until a dedicated validated VDN refine contract exists. '
            'MMH3H3RefineLoRAs now supports acceleration_policy=drop for that future/dedicated path.'
        )
    optimization_inputs = {'attention': selected_attention.pop('attention', 'Default'),
                           'fp16_accumulation': fp16_accumulation, 'sage_attention': sage_attention,
                           'sage_allow_compile': sage_allow_compile}
    optimization_inputs.update({'attention.' + key: value for key, value in selected_attention.items()})
    recipes, summary = validate_sources(packets, width, height, denoise)
    pcm_windows = source_audio_windows(packets, keep_source_audio)
    summary += ' · source PCM audio' if pcm_windows is not None else ' · decoded source audio latent'
    if explicit_sigmas:
        summary += f' · manual sigmas: {len(explicit_sigmas) - 1} steps'
    elif steps_override:
        summary += f' · upscale steps: {steps_override}'
    for family in {r.task_family for r in recipes}:
        model = models.get(family)
        if model is None:
            raise MMH3ResourceError(f'Connect the {family.upper()} base model required by the source segments')
        if getattr(model, 'patches', None) or getattr(model, 'object_patches', None):
            raise MMH3ResourceError('Connect an unpatched base model; source LoRAs are reapplied automatically')
    if graph_builder_factory is None:
        from comfy_execution.graph_utils import GraphBuilder
        graph_builder_factory = GraphBuilder
    graph = graph_builder_factory()
    high = []
    for packet, recipe in zip(packets, recipes):
        effective_steps = len(explicit_sigmas) - 1 if explicit_sigmas else steps_override or recipe.steps
        recipe.info['refine'].update(steps=effective_steps)
        if explicit_sigmas:
            recipe.info['refine'].update(policy='manual_sigmas', sigmas=explicit_sigmas,
                                       denoise=None, max_denoise=None)
        elif steps_override:
            recipe.info['refine']['policy'] = 'override_steps_source_scheduler_tail'
        recipe.info['overrides'] = dict(steps=steps_override, manual_sigmas=explicit_sigmas)
        # HR packets have new latent revisions. The original low-resolution
        # packets remain authoritative for final PCM ownership and chronology.
        high_source = packet
        source_context = packet.manifest.get('extensions', {}).get('minimax_h3', {})
        if source_context.get('audio_timeline') or source_context.get('audio_preservation', {}).get('enabled'):
            high_source = packet.set_extension_value('minimax_h3', 'audio_timeline', None)
            high_source = high_source.set_extension_value('minimax_h3', 'audio_preservation', {'enabled': False})
        prepare = graph.node('MMH3H3LatentUpscalePrepare', packet=high_source, geometry_mode='target_dimensions',
                             scale=1.0, target_width=width, target_height=height, target_megapixels=0.0,
                             align=32, enable_chunking=True)
        upscale = graph.node(UPSCALER_NODE, **build_upscaler_inputs(
            upscaler_api, latent=prepare.out(1), model_name=upscaler_model,
            target_width=width, target_height=height, align=32, device='cuda', precision='bf16',
            offload_after_upscale=force_unload))
        loras = graph.node('MMH3H3RefineLoRAs', packet=prepare.out(0), model=models[recipe.task_family],
                           clip=clip, unknown_policy='error', missing_policy='error', turbo_loras_json=turbo_loras_json)
        optimized = graph.node('MMH3H3ModelOptimizations', model=loras.out(0),
                               **optimization_inputs)
        condition = graph.node('MMH3H3AutoCondition', packet=prepare.out(0), prompt_override='', seed_override=-1,
                               clip=loras.out(1), video_vae=video_vae, audio_vae=audio_vae,
                               width_override=width, height_override=height, frames_override=prepare.out(5))
        report = graph.node('MMH3H3LatentUpscaleReport', geometry_report_json=prepare.out(8),
                            source_lora_report_json=loras.out(4), process_loras_json='[]',
                            tile_plan_json='', tile_run_report_json='', native_tile_adapter_json='',
                            external_tile_report_json='', upscaler_model=upscaler_model,
                            device='cuda', precision='bf16', sigma_profile='source_aware',
                            refine_sampling_json=json.dumps(recipe.info))
        target = graph.node('MMH3H3LatentStitchUpscaleTarget', source_packet=prepare.out(0),
                            upscaled_video_latent=upscale.out(0), source_audio_latent=prepare.out(2),
                            upscale_process_info_json=report.out(0), conditioning_info_json=condition.out(6),
                            **({'previous_high_packet': high[-1]} if high else {}))
        shifted = graph.node('MiniMaxH3SigmaShift', model=optimized.out(0), shift_video=recipe.video_shift,
                             shift_audio=recipe.audio_shift)
        guider = graph.node('BasicGuider', model=shifted.out(0), conditioning=condition.out(0))
        noise = graph.node('RandomNoise', noise_seed=condition.out(2))
        sampler = graph.node('KSamplerSelect', sampler_name=recipe.sampler)
        if explicit_sigmas:
            sigma_input = torch.tensor(explicit_sigmas, dtype=torch.float32)
        else:
            sigmas = graph.node('MMH3H3Scheduler', model=shifted.out(0), scheduler=recipe.scheduler,
                                steps=effective_steps, denoise=recipe.denoise)
            sigma_input = sigmas.out(0)
        sampled = graph.node('SamplerCustomAdvanced', noise=noise.out(0), guider=guider.out(0),
                             sampler=sampler.out(0), sigmas=sigma_input, latent_image=target.out(0))
        restored = graph.node('MMH3H3UpscaleAVRestore', sampled=sampled.out(0), target=target.out(0))
        packed = graph.node('MMH3PackH3Result', packet=prepare.out(0), latent=restored.out(0),
                            operation=target.out(1), mode=target.out(2), status=target.out(3),
                            process_info_json=target.out(4), latent_origin='sampler_output',
                            applied_loras_json=loras.out(5), optimization_profile_json=optimized.out(1))
        high.append(packed.out(0))
    stitch = graph.node('MMH3H3LatentStitch', **{f'segments.segment_{i}': value for i, value in enumerate(high, 1)})
    if pcm_windows is not None:
        audio = graph.node('MMH3H3StitchUpscaleAudio', samples=stitch.out(1), assembly_report_json=stitch.out(3),
                           **{f'segments.segment_{i}': packet for i, packet in enumerate(packets, 1)})
        final_report = audio.out(1)
    else:
        audio = graph.node('VAEDecodeAudio', samples=stitch.out(1), vae=audio_vae)
        final_report = stitch.out(3)
    movie = graph.node('MMH3H3DecodeVideo', samples=stitch.out(1), vae=video_vae,
                       decode_mode=decode_mode, trt_decoder=trt_decoder, streaming=streaming, audio=audio.out(0))
    packed = graph.node('MMH3PackH3Result', packet=stitch.out(0), latent=stitch.out(1), video=movie.out(0),
                        audio=audio.out(0), operation='latent_stitch_decode', mode='h3_continuation',
                        status=stitch.out(2), process_info_json=final_report, latent_origin='derived',
                        optimization_profile_json=optimized.out(1))
    return (packed.out(0), movie.out(0), stitch.out(1), final_report), graph.finalize(), summary
