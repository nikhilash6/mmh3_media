"""Persistent PCM clocks for protected F01 -> F02 continuations."""
from __future__ import annotations

from copy import deepcopy
import math
from uuid import uuid4

from .audio_preservation import canonical_audio, checked_audio, fit_audio_samples, select_source_audio, protect_audio_target
from .errors import MMH3ResourceError
from .h3 import nested_parts, make_nested_tensor, validate_h3_av_latent
from .resource_model import descriptor_from_media_metadata
from .media_metadata import describe_media_payload
from .stitch import pcm_boundary

CONTRACT = "mmh3_audio_timeline_v1"


def _context(packet):
    return packet.manifest.get("extensions", {}).get("minimax_h3", {})


def timeline_from_packet(packet):
    record = _context(packet).get("audio_timeline")
    if record is None:
        return None
    if (not isinstance(record, dict) or record.get("contract") != CONTRACT
            or record.get("fps") != 24 or record.get("sample_rate") != 32000):
        raise MMH3ResourceError("Unknown protected audio timeline; load a supported F01/F02 packet")
    start, end, context = (record.get(k) for k in ("segment_start_frame", "segment_end_frame", "context_frames"))
    if (any(type(v) is not int for v in (start, end, context)) or start < 0 or end <= start
            or not 0 <= context < end - start or not isinstance(record.get("sequence_id"), str)
            or not record["sequence_id"] or not isinstance(record.get("tracks"), list) or not record["tracks"]):
        raise MMH3ResourceError("Invalid protected audio timeline bounds")
    previous_end = 0
    for track in record["tracks"]:
        if (not isinstance(track, dict) or not isinstance(track.get("resource_id"), str)
                or not isinstance(track.get("revision"), str)
                or type(track.get("start_sample")) is not int or track["start_sample"] < previous_end
                or type(track.get("source_offset")) is not int or track["source_offset"] < 0
                or (track.get("end_sample") is not None and
                    (type(track["end_sample"]) is not int or track["end_sample"] < track["start_sample"]))):
            raise MMH3ResourceError("Invalid or overlapping audio timeline tracks")
        previous_end = track["end_sample"] if track.get("end_sample") is not None else float("inf")
    return deepcopy(record)


def _put_audio(packet, audio, name):
    return packet.put_with_id(audio, kind="audio", role="auxiliary", name=name,
        descriptor=descriptor_from_media_metadata("audio", describe_media_payload(audio, "audio")))


def _track(packet, audio, start_sample, source_offset=0):
    primary = packet.get_primary("audio")
    rid = next((r["id"] for r in packet.manifest["resources"] if r["kind"] == "audio"
                and (not primary or r["id"] != primary["id"])
                and isinstance(packet.payloads.get(r["id"]), dict)
                and packet.payloads[r["id"]].get("waveform") is audio["waveform"]
                and packet.payloads[r["id"]].get("sample_rate") == 32000), None)
    if rid is None:
        packet, rid = _put_audio(packet, audio, "Full audio timeline source")
    resource = packet.get_by_id(rid)
    return packet, {"resource_id": rid, "revision": resource["content"]["revision"],
                    "start_sample": start_sample, "source_offset": source_offset,
                    "end_sample": None}


def initialize_audio_timeline(packet, frames):
    """Seed the clock without losing the full source beyond the first clip.

    A stretched first clip has consumed its entire source. Its mapped waveform
    ends at that clip; continuation must never stretch/replay the song again.
    """
    report = _context(packet).get("audio_preservation", {})
    if not report.get("enabled"):
        return packet
    primary = packet.get_primary("audio")
    if primary is None:
        raise MMH3ResourceError("Protected audio is missing")
    if report.get("fit") == "stretch":
        audio = packet.ref(primary["id"]).materialize()
    elif report.get("resource_id"):
        selected, _ = select_source_audio(packet, report["resource_id"])
        audio = canonical_audio(selected)
    else:
        audio = packet.ref(primary["id"]).materialize()
    packet, track = _track(packet, audio, 0)
    record = {"contract": CONTRACT, "sequence_id": str(uuid4()), "fps": 24, "sample_rate": 32000,
              "segment_start_frame": 0, "segment_end_frame": frames, "context_frames": 0,
              "tracks": [track]}
    return packet.set_extension_value("minimax_h3", "audio_timeline", record)


def bind_audio_timeline(packet, output_ids, frames=None):
    """Bind the cursor to the actual committed result, never to a queued draft."""
    record = timeline_from_packet(packet)
    if record is None:
        return packet
    if frames is not None and record["segment_end_frame"] - record["segment_start_frame"] != frames:
        raise MMH3ResourceError("Protected audio timeline length differs from the sampled video")
    for kind in ("latent", "audio"):
        rid = output_ids.get(kind)
        if rid:
            resource = packet.get_by_id(rid)
            record[kind + "_binding"] = {"resource_id": rid, "revision": resource["content"]["revision"]}
    return packet.set_extension_value("minimax_h3", "audio_timeline", record)


def render_timeline_audio(packet, record, start_frame, end_frame):
    """Half-open absolute sample windows avoid accumulating 24/32k rounding."""
    start, end = pcm_boundary(start_frame), pcm_boundary(end_frame)
    if start < 0 or end <= start:
        raise MMH3ResourceError("Invalid audio timeline window")
    from .pcm_storage import allocate_pcm, copy_pcm
    output = allocate_pcm(2, end - start)
    output.zero_()
    for track in record["tracks"]:
        resource = packet.get_by_id(track["resource_id"])
        if resource is None or resource.get("content", {}).get("revision") != track["revision"]:
            raise MMH3ResourceError("Audio timeline source changed or was removed")
        track_start = int(track["start_sample"])
        if track_start >= end or (track.get("end_sample") is not None and track["end_sample"] <= start):
            continue
        audio = packet.ref(resource["id"]).materialize()
        wave, rate = checked_audio(audio)
        if rate != 32000 or wave.shape[1] != 2:
            raise MMH3ResourceError("Audio timeline sources must be 32 kHz stereo")
        offset = int(track["source_offset"])
        track_end = track_start + max(0, wave.shape[-1] - offset)
        if track.get("end_sample") is not None:
            track_end = min(track_end, int(track["end_sample"]))
        left, right = max(start, track_start), min(end, track_end)
        if right > left:
            source_start = offset + left - track_start
            copy_pcm(output, left - start, wave[..., source_start:source_start + right - left])
    return {"waveform": output, "sample_rate": 32000}


def prepare_audio_continuation(packet, latent, audio_vae, handover, *, mode="Continue source",
                               new_audio=None, audio_source="auto", new_audio_fit="cut", new_audio_offset_seconds=0.0):
    if mode not in {"Generate", "Continue source", "New audio"}:
        raise MMH3ResourceError("Choose Generate, Continue source or New audio")
    if mode == "Generate":
        out = packet.set_extension_value("minimax_h3", "audio_preservation", {"enabled": False})
        return latent, out.set_extension_value("minimax_h3", "audio_timeline", None), "Generated continuation audio"
    info = validate_h3_av_latent(latent, strict_audio_length=True)
    if info.frames is None:
        raise MMH3ResourceError("Protected continuation requires the native H3 video clock")
    context = handover.get("video_handover_frames")
    if (handover.get("contract") != "minimax_h3_joint_av_continuation_v1" or type(context) is not int
            or context < 1 or context >= info.frames or handover.get("target_frames") != info.frames
            or handover.get("audio_handover_frames") != context or handover.get("audio_feather_frames", 0) != 0):
        raise MMH3ResourceError("Protected continuation requires equal AV handovers, no audio feather, and new frames after the context")
    if handover.get("copied_audio_t") != context * 40 // 24 or context * 40 % 24:
        raise MMH3ResourceError("Protected audio context must occupy exact native audio ticks")
    source = packet.get_primary("latent")
    parent_audio = packet.get_primary("audio")
    parent_audio_binding = ({"resource_id": parent_audio["id"], "revision": parent_audio["content"]["revision"]}
                            if parent_audio else None)
    if (source is None or source["id"] != handover.get("source_latent_resource_id")
            or source["content"]["revision"] != handover.get("source_latent_revision")
            or packet.manifest["id"] != handover.get("source_packet_id")):
        raise MMH3ResourceError("Continuation source changed after handover")
    record = timeline_from_packet(packet)
    if record is None:
        previous = packet.manifest.get("extensions", {}).get("mmh3_media", {}).get("last_process", {})
        if previous.get("operation") not in {None, "", "generate"}:
            raise MMH3ResourceError("This source has no original audio clock; start from its F01 packet or use Generate")
        # Upgrade an existing F01 archive: prefer its full preserved source.
        if not _context(packet).get("audio_preservation", {}).get("enabled"):
            from .audio_preservation import prepare_audio_packet
            packet, report = prepare_audio_packet(packet, int(handover["source_frames"]))
            if report["source"] == "none":
                raise MMH3ResourceError("Parent has no source PCM soundtrack; load an F01 archive with audio before protecting its continuation")
        packet = initialize_audio_timeline(packet, int(handover["source_frames"]))
        packet = bind_audio_timeline(packet, {"latent": source["id"], "audio": packet.get_primary("audio")["id"]}, int(handover["source_frames"]))
        record = timeline_from_packet(packet)
    binding = record.get("latent_binding", {})
    current_audio = packet.get_primary("audio")
    if (binding.get("resource_id") != source["id"] or binding.get("revision") != source["content"]["revision"]
            or record["segment_end_frame"] - record["segment_start_frame"] != handover.get("source_frames")
            or current_audio is None or record.get("audio_binding") !=
                {"resource_id": current_audio["id"], "revision": current_audio["content"]["revision"]}):
        raise MMH3ResourceError("Audio clock does not match the parent latent; load the committed parent packet")
    parent_end = record["segment_end_frame"]
    start = parent_end - context
    end = start + info.frames
    parent_binding = deepcopy(binding)
    if mode == "Continue source" and new_audio is not None:
        raise MMH3ResourceError("New audio is connected: select New audio to place it at the next visible frame")
    if mode == "New audio":
        if new_audio is None and audio_source == "auto":
            raise MMH3ResourceError("New audio requires the new_audio input or an explicit audio/video source")
        selected, _ = select_source_audio(packet, audio_source, new_audio)
        if selected is None:
            raise MMH3ResourceError("No new soundtrack was selected")
        audio = canonical_audio(selected)
        offset_seconds = float(new_audio_offset_seconds)
        if not math.isfinite(offset_seconds) or offset_seconds < 0:
            raise MMH3ResourceError("New audio offset must be finite and nonnegative")
        offset = round(offset_seconds * 32000)
        if offset >= audio["waveform"].shape[-1]:
            raise MMH3ResourceError("New audio offset is past the end of its source")
        # Preserve the full original, including any skipped lead-in.
        packet, _ = _put_audio(packet, selected, "Original new soundtrack")
        if new_audio_fit == "stretch":
            audio = fit_audio_samples({"waveform": audio["waveform"][..., offset:], "sample_rate": 32000},
                                      pcm_boundary(end) - pcm_boundary(parent_end), "stretch")
            offset = 0
        elif new_audio_fit != "cut":
            raise MMH3ResourceError("New audio fit must be cut or stretch")
        boundary = pcm_boundary(parent_end)
        for track in record["tracks"]:
            if track.get("end_sample") is None or track["end_sample"] > boundary:
                track["end_sample"] = boundary
        packet, track = _track(packet, audio, boundary, offset)
        record["tracks"].append(track)
    record.update(segment_start_frame=start, segment_end_frame=end, context_frames=context,
                  parent_latent_binding=parent_binding, parent_audio_binding=parent_audio_binding,
                  parent_packet_id=packet.manifest["id"], parent_end_frame=parent_end)
    # These bindings describe the parent until Pack commits the new result.
    record.pop("latent_binding", None)
    record.pop("audio_binding", None)
    audio = render_timeline_audio(packet, record, start, end)
    packet, rid = _put_audio(packet, audio, "Protected continuation soundtrack")
    packet = packet.set_primary("audio", rid)
    report = {"contract": "mmh3_audio_preservation_v1", "enabled": True, "fit": "timeline",
              "source": mode, "frames": info.frames, "output_resource_id": rid,
              "output_samples": audio["waveform"].shape[-1], "sample_rate": 32000,
              "channels": 2, "generated_audio": "excluded", "timeline_start_frame": start, "timeline_end_frame": end}
    packet = packet.set_extension_value("minimax_h3", "audio_preservation", report)
    packet = packet.set_extension_value("minimax_h3", "audio_timeline", record)
    target = protect_audio_target(latent, audio, audio_vae)
    video, encoded = nested_parts(target["samples"])
    copied_ticks = int(handover["copied_audio_t"])
    original_audio = nested_parts(latent["samples"])[1]
    encoded = encoded.clone()
    encoded[..., :copied_ticks] = original_audio[..., :copied_ticks].to(encoded)
    target = {**target, "samples": make_nested_tensor([video, encoded])}
    return target, packet, f"{mode} · timeline {start/24:.3f}–{end/24:.3f} s · {(end-parent_end)/24:.3f} s new"


def timeline_stitch_windows(packets):
    """Use the recorded ownership seam; generic montages retain their old rules."""
    records = [timeline_from_packet(p) for p in packets]
    if not any(r and r.get("context_frames", 0) for r in records):
        return None
    if records[0] is None and len(records) > 1 and records[1] is not None:
        # A pre-update F01 archive can be the parent of an upgraded F02.
        child = records[1]
        frames = packets[0].manifest.get("generation", {}).get("frames")
        if child.get("parent_end_frame") == frames:
            records[0] = {**child, "segment_start_frame": 0, "segment_end_frame": frames,
                          "context_frames": 0, "latent_binding": deepcopy(child["parent_latent_binding"])}
            audio = packets[0].get_primary("audio")
            if audio:
                records[0]["audio_binding"] = {"resource_id": audio["id"], "revision": audio["content"]["revision"]}
    if any(r is None for r in records):
        raise MMH3ResourceError("Protected audio continuation assembly requires the complete, ordered F01/F02 timeline")
    windows = []
    for index, (packet, record) in enumerate(zip(packets, records)):
        latent = packet.get_primary("latent")
        binding = record.get("latent_binding", {})
        if latent is None or binding != {"resource_id": latent["id"], "revision": latent["content"]["revision"]}:
            raise MMH3ResourceError(f"Audio timeline segment {index} no longer matches its sampled latent")
        audio = packet.get_primary("audio")
        if audio is None or record.get("audio_binding") != {"resource_id": audio["id"], "revision": audio["content"]["revision"]}:
            raise MMH3ResourceError("Audio timeline soundtrack was replaced after generation")
        start, end = record["segment_start_frame"], record["segment_end_frame"]
        if end - start != packet.manifest.get("generation", {}).get("frames"):
            raise MMH3ResourceError("Audio timeline frame count changed after generation")
        trim = 0
        if index:
            previous = records[index - 1]
            previous_audio = packets[index - 1].get_primary("audio")
            if (record["sequence_id"] != previous["sequence_id"]
                    or record.get("parent_packet_id") != packets[index - 1].manifest["id"]
                    or record.get("parent_latent_binding") != previous["latent_binding"]
                    or record.get("parent_audio_binding") != {"resource_id": previous_audio["id"], "revision": previous_audio["content"]["revision"]}
                    or record.get("parent_end_frame") != previous["segment_end_frame"]
                    or start + record["context_frames"] != previous["segment_end_frame"]):
                raise MMH3ResourceError("Audio timeline segments are missing, reordered, or belong to different takes")
            trim = record["context_frames"]
        windows.append({"trim_frames": trim, "trim_samples": pcm_boundary(start + trim) - pcm_boundary(start),
                        "raw_start_frame": start, "end_frame": end})
    return windows
