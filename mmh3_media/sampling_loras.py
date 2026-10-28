"""Restore source adapters before sampling, including CLIP and provenance."""
import hashlib

from .errors import MMH3ResourceError
from .lora_provenance import get_generation_loras
from .lora_reapply import build_high_sigma_lora_plan
from .turbo_loras import parse_turbo_loras, replace_source_turbo_loras, turbo_lora_mode


def apply_source_sampling_loras(packet, model, clip, selection_json=""):
    import folder_paths
    from nodes import LoraLoader

    selection = parse_turbo_loras(selection_json)
    if selection is not None:
        packet = replace_source_turbo_loras(
            packet, selection, extension=turbo_lora_mode(selection_json) == "extension")
    recorded = get_generation_loras(packet)
    wanted = {entry["name"] for entry in recorded or () if entry.get("reapply_for_high_sigma", True)}
    inventory = {}
    for name in folder_paths.get_filename_list("loras"):
        normalized = name.replace("\\", "/")
        if normalized not in wanted:
            continue
        path = folder_paths.get_full_path("loras", name)
        if path:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            inventory[normalized] = digest.hexdigest()
    plan = build_high_sigma_lora_plan(packet, lora_inventory=inventory)
    if not plan.ready:
        raise MMH3ResourceError("Cannot inherit source LoRAs: " + "; ".join(
            d.message for d in plan.diagnostics if d.severity == "error"))
    if clip is None and any(entry.get("strength_clip") for entry in plan.applied):
        raise MMH3ResourceError("Connect source_clip to H3 Sampling to inherit source CLIP LoRAs")
    applied = []
    for entry in plan.applied:
        model, clip = LoraLoader().load_lora(
            model, clip, entry["name"], entry["strength_model"], entry.get("strength_clip") or 0.0)
        applied.append({**{k: v for k, v in entry.items() if k != "source_index"},
                        "sha256": inventory[entry["name"]]})
    return model, clip, applied
