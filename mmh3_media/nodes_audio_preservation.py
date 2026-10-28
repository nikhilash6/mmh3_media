"""Audio protection at the two boundaries of an existing sampler graph."""
from __future__ import annotations

from .node_support import CATEGORY, MMH3, _packet, io, ui
from .errors import MMH3ResourceError


class MMH3H3AudioPreservationPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3H3AudioPreservationPrepare", category=CATEGORY + "/Internal",
            description="Prepare one soundtrack and freeze its target audio stream before sampling. Off is a latent pass-through.",
            inputs=[MMH3.Input("packet"), io.Latent.Input("latent"), io.Vae.Input("audio_vae"),
                    io.Boolean.Input("preserve_audio", display_name="Keep source audio", default=False,
                                     tooltip="Protect the source soundtrack during sampling and restore it at output. No source produces silence."),
                    io.Combo.Input("audio_fit", options=["cut", "stretch"], default="cut", display_name="Audio fit",
                                   tooltip="cut trims excess or pads a short track with silence. stretch fits the whole track while preserving pitch."),
                    io.String.Input("audio_source", default="auto", display_name="Audio source",
                                    tooltip="auto selects one available soundtrack. Use audio:0 for the first audio reference, video:0 for the first reference video's soundtrack, or a resource ID. Multiple tracks require a choice.")],
            outputs=[io.Latent.Output("latent"), MMH3.Output("packet")])

    @classmethod
    def execute(cls, packet, latent, audio_vae, preserve_audio=False, audio_fit="cut", audio_source="auto"):
        value = _packet(packet)
        if not preserve_audio:
            value = value.set_extension_value("minimax_h3", "audio_preservation", {"enabled": False})
            return io.NodeOutput(latent, value)
        from .nodes_editing import require_joint_masks
        from .audio_preservation import prepare_audio_packet, protect_audio_target
        from .h3 import validate_h3_av_latent
        require_joint_masks()
        info = validate_h3_av_latent(latent, strict_audio_length=True)
        value, report = prepare_audio_packet(value, info.frames, fit=audio_fit, source=audio_source)
        from .audio_timeline import initialize_audio_timeline
        value = initialize_audio_timeline(value, info.frames)
        audio = value.ref(value.get_primary("audio")["id"]).materialize()
        protected = protect_audio_target(latent, audio, audio_vae)
        summary = ("No source soundtrack · silent output" if report["source"] == "none"
                   else f"Source audio protected · {report['source']} · {audio_fit}")
        return io.NodeOutput(protected, value, ui=ui.PreviewText(summary))


class MMH3H3AudioPreservationFinish(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3H3AudioPreservationFinish", category=CATEGORY + "/Internal",
            description="Restore protected latent values and deliver source PCM; off uses the existing audio decoder.",
            inputs=[MMH3.Input("packet"), io.Latent.Input("protected_source"),
                    io.Latent.Input("sampled"), io.Vae.Input("audio_vae")],
            outputs=[io.Latent.Output("latent"), io.Audio.Output("audio")], enable_expand=True)

    @classmethod
    def execute(cls, packet, protected_source, sampled, audio_vae):
        value = _packet(packet)
        report = value.manifest.get("extensions", {}).get("minimax_h3", {}).get("audio_preservation", {})
        if report.get("enabled"):
            from .av_edit import restore_av_protection
            resource = value.get_primary("audio")
            if resource is None or resource["id"] != report.get("output_resource_id"):
                raise MMH3ResourceError("Protected soundtrack changed after preparation; prepare generation again")
            audio = value.ref(resource["id"]).materialize()
            return io.NodeOutput(restore_av_protection(protected_source, sampled), audio)
        from comfy_execution.graph_utils import GraphBuilder
        graph = GraphBuilder()
        decoded = graph.node("VAEDecodeAudio", samples=sampled, vae=audio_vae)
        return io.NodeOutput(sampled, decoded.out(0), expand=graph.finalize())


class MMH3H3AudioContinuationPrepare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MMH3H3AudioContinuationPrepare", category=CATEGORY + "/Internal",
            inputs=[MMH3.Input("packet"), io.Latent.Input("latent"), io.Vae.Input("audio_vae"),
                    io.String.Input("handover_info_json", force_input=True),
                    io.Combo.Input("audio_mode", options=["Continue source", "New audio", "Generate"], default="Continue source", display_name="Audio mode",
                                   tooltip="Continue source reads the next window of the saved full track. New audio starts after the inherited context. Generate uses H3 audio."),
                    io.Combo.Input("new_audio_fit", options=["cut", "stretch"], default="cut", display_name="New audio fit",
                                   tooltip="Only New audio: cut retains normal speed, stretch fits the full new track to this segment's new frames. Existing timeline speed never changes."),
                    io.String.Input("audio_source", default="auto", display_name="New audio source",
                                    tooltip="Only New audio: connect new_audio, or select audio:0 / video:0 / a resource ID. An inherited source is never chosen implicitly here."),
                    io.Float.Input("new_audio_offset_seconds", default=0.0, min=0.0, max=86400.0, step=0.01, display_name="New audio start (s)", advanced=True,
                                   tooltip="Start position within the new audio file. Placement on the output timeline is automatic, after the inherited context."),
                    io.Audio.Input("new_audio", optional=True)],
            outputs=[io.Latent.Output("latent"), MMH3.Output("packet")])

    @classmethod
    def execute(cls, packet, latent, audio_vae, handover_info_json, audio_mode="Continue source", new_audio_fit="cut",
                audio_source="auto", new_audio_offset_seconds=0.0, new_audio=None):
        import json
        from .audio_timeline import prepare_audio_continuation
        if audio_mode != "Generate":
            from .nodes_editing import require_joint_masks
            require_joint_masks()
        target, value, summary = prepare_audio_continuation(_packet(packet), latent, audio_vae,
            json.loads(handover_info_json), mode=audio_mode, new_audio_fit=new_audio_fit,
            audio_source=audio_source, new_audio_offset_seconds=new_audio_offset_seconds, new_audio=new_audio)
        return io.NodeOutput(target, value, ui=ui.PreviewText(summary))
