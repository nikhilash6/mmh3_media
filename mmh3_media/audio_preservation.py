"""Choose one soundtrack, fit it once, and keep it outside the generated mix."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .errors import MMH3ResourceError


def checked_audio(audio):
    if not isinstance(audio, dict):
        raise MMH3ResourceError("Source audio must contain waveform and sample_rate")
    wave, rate = audio.get("waveform"), audio.get("sample_rate")
    if (not isinstance(wave, torch.Tensor) or wave.ndim != 3 or wave.shape[0] != 1
            or wave.shape[1] not in (1, 2) or wave.shape[-1] < 1
            or not wave.is_floating_point() or not torch.isfinite(wave).all()
            or type(rate) is not int or rate <= 0):
        raise MMH3ResourceError("Source audio must be finite mono/stereo [1,channels,samples] with a positive sample rate")
    return wave, rate


def select_source_audio(packet, source="auto", audio=None):
    """Explicit input > selected slot/id > primary > active audio > video sound.

    Independent tracks are never mixed or arbitrarily selected. Historical
    auxiliary resources are considered only when explicitly selected/primary.
    """
    if audio is not None:
        checked_audio(audio)
        return audio, {"source": "audio_input"}
    if not isinstance(source, str) or not source.strip():
        raise MMH3ResourceError("Choose an audio source: auto, audio:0, video:0, or a resource ID")
    source = source.strip()
    resources = packet.manifest["resources"]
    active = [r for r in resources if r["kind"] in {"audio", "video"} and r.get("role") == "reference"
              and r.get("extensions", {}).get("minimax_h3", {}).get("reference", {}).get("enabled", True)]
    from .reference_cards import reference_slots
    slots = {key: r for key, r in reference_slots(packet)}

    def read(resource):
        # A separate paired soundtrack is authoritative over an embedded one.
        if resource["kind"] == "video":
            paired = [r for r in active if r["kind"] == "audio" and
                      r.get("extensions", {}).get("minimax_h3", {}).get("reference", {}).get("binding", {}).get("video_resource_id") == resource["id"]]
            if len(paired) > 1:
                ambiguous(paired)
            if paired:
                return read(paired[0])
        payload = packet.ref(resource["id"]).materialize()
        result = payload if resource["kind"] == "audio" else getattr(payload.get_components(), "audio", None)
        if result is not None:
            checked_audio(result)
        key = next((k for k, r in slots.items() if r["id"] == resource["id"]), resource["id"])
        return result, {"source": key, "resource_id": resource["id"], "kind": resource["kind"]}

    def ambiguous(candidates):
        choices = [next((k for k, r in slots.items() if r["id"] == c["id"]), c["id"]) for c in candidates]
        raise MMH3ResourceError("Multiple source soundtracks; choose Audio source: " + ", ".join(choices)
                                + ". They will not be mixed automatically.")

    if source != "auto":
        resource = slots.get(source) or next((r for r in resources if r["id"] == source), None)
        if resource is None or resource["kind"] not in {"audio", "video"}:
            raise MMH3ResourceError(f"Audio source {source!r} is unavailable; choose auto or an existing audio/video slot")
        result, report = read(resource)
        if result is None:
            raise MMH3ResourceError(f"Selected audio source {source!r} has no soundtrack")
        return result, report
    # Refit the original on a new run, rather than stretching an already fitted
    # output repeatedly. The binding applies only to this protected output slot.
    saved = packet.manifest.get("extensions", {}).get("minimax_h3", {}).get("audio_preservation", {})
    primary = packet.get_primary("audio")
    if saved.get("enabled") and primary and primary["id"] == saved.get("output_resource_id"):
        original = next((r for r in resources if r["id"] == saved.get("resource_id")), None)
        if original is not None:
            result, report = read(original)
            if result is not None:
                return result, report
    for kind in ("audio", "video"):
        resource = packet.get_primary(kind)
        if resource is not None:
            result, report = read(resource)
            if result is not None:
                return result, report
    audios = [r for r in active if r["kind"] == "audio"]
    if len(audios) > 1:
        ambiguous(audios)
    if audios:
        return read(audios[0])
    videos = []
    for resource in active:
        if resource["kind"] == "video":
            result, report = read(resource)
            if result is not None:
                videos.append((resource, result, report))
    if len(videos) > 1:
        ambiguous([item[0] for item in videos])
    if videos:
        return videos[0][1:]
    return None, {"source": "none", "reason": "No source soundtrack; protected output is silent"}


def _stretch(wave, samples):
    """Phase-vocoder tempo change; rate metadata and pitch stay unchanged."""
    from torchaudio.functional import phase_vocoder
    # CPU float32 also supports hosts with half-precision source tensors.
    original_device, original_dtype = wave.device, wave.dtype
    flat = wave.detach().to(device="cpu", dtype=torch.float32).flatten(0, 1)
    n_fft, hop = 1024, 256
    window = torch.hann_window(n_fft)
    spec = torch.stft(flat, n_fft, hop_length=hop, window=window, return_complex=True, pad_mode="constant")
    phase = torch.linspace(0, torch.pi * hop, spec.shape[-2])[..., None]
    shifted = phase_vocoder(spec, wave.shape[-1] / samples, phase)
    result = torch.istft(shifted, n_fft, hop_length=hop, window=window, length=samples)
    return result.reshape(1, wave.shape[1], samples).to(device=original_device, dtype=original_dtype)


def fit_source_audio(audio, frames, fit="cut"):
    if type(frames) is not int or frames < 1 or fit not in {"cut", "stretch"}:
        raise MMH3ResourceError("Audio fit requires positive frames and cut or stretch")
    if audio is None:
        return {"waveform": torch.zeros(1, 2, round(frames * 32000 / 24)), "sample_rate": 32000}
    wave, rate = checked_audio(audio)
    samples = round(frames * rate / 24)
    return fit_audio_samples(audio, samples, fit)


def fit_audio_samples(audio, samples, fit="cut"):
    wave, rate = checked_audio(audio)
    if type(samples) is not int or samples < 1 or fit not in {"cut", "stretch"}:
        raise MMH3ResourceError("Audio fit requires positive samples and cut or stretch")
    if wave.shape[-1] == samples:
        return audio
    if fit == "cut":
        wave = F.pad(wave[..., :samples], (0, max(0, samples - wave.shape[-1])))
    else:
        wave = _stretch(wave, samples)
    result = {"waveform": wave, "sample_rate": rate}
    checked_audio(result)
    return result


def canonical_audio(audio):
    wave, rate = checked_audio(audio)
    if rate != 32000:
        from torchaudio.functional import resample
        wave = resample(wave.to(dtype=torch.float32), rate, 32000)
    if wave.shape[1] == 1:
        wave = wave.repeat(1, 2, 1)
    return {"waveform": wave, "sample_rate": 32000}


def prepare_preserved_audio(packet, frames, *, fit="cut", source="auto", audio=None):
    selected, report = select_source_audio(packet, source, audio)
    model_audio = canonical_audio(selected) if selected is not None else None
    fitted = fit_source_audio(model_audio, frames, fit)
    report.update(contract="mmh3_audio_preservation_v1", enabled=True, fit=fit, frames=frames,
                  source_samples=int(selected["waveform"].shape[-1]) if selected else 0,
                  source_sample_rate=selected["sample_rate"] if selected else None,
                  output_samples=int(fitted["waveform"].shape[-1]), sample_rate=fitted["sample_rate"],
                  channels=int(fitted["waveform"].shape[1]), generated_audio="excluded")
    return fitted, report


def prepare_audio_packet(packet, frames, **options):
    """Keep original resources/aliases intact; generated output owns a new slot."""
    from .media_metadata import describe_media_payload
    from .resource_model import descriptor_from_media_metadata
    # A directly connected AUDIO must remain available after the input node is
    # gone, including the discarded tail in cut mode.
    explicit = options.get("audio")
    if explicit is not None:
        checked_audio(explicit)
        packet, source_id = packet.put_with_id(explicit, kind="audio", role="auxiliary", name="Source soundtrack",
            descriptor=descriptor_from_media_metadata("audio", describe_media_payload(explicit, "audio")))
        options = {**options, "audio": None, "source": source_id}
    audio, report = prepare_preserved_audio(packet, frames, **options)
    out, resource_id = packet.put_with_id(audio, kind="audio", role="auxiliary", name="Protected soundtrack",
        descriptor=descriptor_from_media_metadata("audio", describe_media_payload(audio, "audio")))
    out = out.set_primary("audio", resource_id)
    report["output_resource_id"] = resource_id
    return out.set_extension_value("minimax_h3", "audio_preservation", report), report


def protect_audio_target(latent, audio, audio_vae):
    """Use a separate 32 kHz/stereo model copy; never mutate delivery PCM."""
    from .h3 import validate_h3_av_latent, nested_parts, make_nested_tensor
    from .audio_vae import preserve_h3_audio_onset
    info = validate_h3_av_latent(latent, strict_audio_length=True)
    video, target = nested_parts(latent["samples"])
    wave, rate = checked_audio(audio)
    wave = wave.to(dtype=torch.float32)
    if rate != 32000:
        from torchaudio.functional import resample
        wave = resample(wave, rate, 32000)
    if wave.shape[1] == 1:
        wave = wave.repeat(1, 2, 1)
    samples = target.shape[-1] * 800
    wave = F.pad(wave[..., :samples], (0, max(0, samples - wave.shape[-1])))
    encoded = preserve_h3_audio_onset(audio_vae).encode(wave.movedim(1, -1))
    if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != info.audio_shape or not torch.isfinite(encoded).all():
        raise MMH3ResourceError("Audio VAE returned an incompatible protected target; generation stopped")
    encoded = encoded.to(target)
    video_mask = torch.ones_like(video)
    if latent.get("noise_mask") is not None:
        masks = nested_parts(latent["noise_mask"])
        if len(masks) != 2 or tuple(masks[0].shape) != tuple(video.shape):
            raise MMH3ResourceError("Invalid existing AV protection mask")
        video_mask = masks[0]
    return {**latent, "samples": make_nested_tensor([video, encoded]),
            "noise_mask": make_nested_tensor([video_mask, torch.zeros_like(encoded)])}
