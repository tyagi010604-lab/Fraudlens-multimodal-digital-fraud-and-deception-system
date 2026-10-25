"""Module 7: audio-video / cross-module consistency checks (transparent rules only).

Implemented checks compare *available* evidence. NO lip-sync analysis is implemented, and this module
never claims that lip movements match or mismatch speech (reported as an explicit "not assessed" item).
"""
import re

import config


def _wps(text, duration):
    words = len(re.findall(r"\w+", text))
    return words, (words / duration if duration > 0 else None)


def run_consistency(ctx):
    """ctx keys (all optional): video, audio, audio_source ('separate'|'extracted'|None), face, voice,
    transcript (str), video_has_audio_track (bool|None)."""
    video, audio = ctx.get("video"), ctx.get("audio")
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    both = video is not None and audio is not None and audio.get("audio_info") is not None and \
        video.get("video_meta") is not None
    # C1 duration agreement (only meaningful when audio was supplied separately)
    if both and ctx.get("audio_source") in ("separate", "extracted"):
        dv = video["video_meta"].get("duration_s")
        da = audio["audio_info"].get("original_duration_s") or audio["audio_info"]["analysed_duration_s"]
        # extracted audio is cut to AUDIO_MAX_SECONDS, so only compare when the video is not longer than that
        if dv and da and not (ctx.get("audio_source") == "extracted" and dv > config.AUDIO_MAX_SECONDS):
            diff = abs(dv - da)
            if diff > 1.0 and diff / max(dv, da) > 0.10:
                add("Audio/video duration agreement", "flagged",
                    f"Video is {dv:.1f}s but the audio is {da:.1f}s (difference {diff:.1f}s). "
                    + ("The audio stream and video stream of this file differ in length." if ctx.get("audio_source") == "extracted"
                       else "The files may not belong together or one was trimmed."))
            else:
                add("Audio/video duration agreement", "passed", f"Video {dv:.1f}s vs audio {da:.1f}s (within tolerance).")
    elif video is not None and ctx.get("audio_source") is None and ctx.get("video_has_audio_track") is False:
        add("Video has an audio track", "not_assessed", "The video has no audio stream, so audio-video checks cannot run "
                                                        "(a silent video is not by itself suspicious).")
    # C2 authenticity-score conflict between specialist modules
    vs, as_ = (video or {}).get("risk_score"), (audio or {}).get("risk_score")
    if vs is not None and as_ is not None and not (video.get("unknown") or audio.get("unknown")):
        if (vs >= 0.65 and as_ < 0.35) or (as_ >= 0.65 and vs < 0.35):
            add("Video vs audio authenticity scores", "flagged",
                f"Video score {vs:.2f} and audio score {as_:.2f} point in opposite directions (one high, one low).")
        else:
            add("Video vs audio authenticity scores", "passed", f"Video {vs:.2f} / audio {as_:.2f}: no strong disagreement.")
    # C3 identity conflict
    f, v = ctx.get("face"), ctx.get("voice")
    if f and v and f.get("label") in ("Match", "Mismatch") and v.get("label") in ("Match", "Mismatch"):
        if f["label"] != v["label"]:
            add("Face vs voice identity", "flagged",
                f"Face check = {f['label']} but voice check = {v['label']} against the supplied references "
                "(voice check is experimental).")
        else:
            add("Face vs voice identity", "passed", f"Both identity checks returned {f['label']}.")
    # C4/C5 transcript vs audio
    tr = (ctx.get("transcript") or "").strip()
    if tr and audio is not None and audio.get("audio_info"):
        info = audio["audio_info"]
        dur = info["analysed_duration_s"]
        words, wps = _wps(tr, dur)
        if info["silence_ratio"] > 0.9 and words >= 5:
            add("Transcript vs audio content", "flagged",
                f"A {words}-word transcript was supplied but {info['silence_ratio']:.0%} of the audio is near-silent.")
        elif wps is not None and words >= 8 and not (0.4 <= wps <= 6.5):
            add("Transcript length vs audio duration", "flagged",
                f"{words} words in {dur:.1f}s = {wps:.1f} words/s, outside the plausible speech range (0.4-6.5). "
                "The transcript may not belong to this recording (transcript accuracy is not verified).")
        elif words >= 8:
            add("Transcript length vs audio duration", "passed", f"{words} words in {dur:.1f}s = {wps:.1f} words/s (plausible).")
    elif tr and audio is None:
        add("Transcript vs audio", "not_assessed", "A transcript was given but no audio was available to compare.")
    # explicit non-claims
    if video is not None and audio is not None:
        add("Lip-sync / audio-visual synchronisation", "not_assessed",
            "Not implemented in FraudLens: no statement is made about whether lip movements match the speech.")
    assessed = [c for c in checks if c["status"] in ("passed", "flagged")]
    flagged = [c for c in assessed if c["status"] == "flagged"]
    if not assessed:
        return {"status": "not_assessed", "label": "Not assessed", "risk_score": None, "checks": checks,
                "reason": "Consistency checks need at least two kinds of evidence (e.g. video + audio, or a transcript + audio)."}
    return {"status": "ready", "label": "Inconsistencies flagged" if flagged else "No inconsistency found by the checks performed",
            "risk_score": round(len(flagged) / len(assessed), 3), "checks": checks,
            "flagged": [c["check"] for c in flagged],
            "note": "Rule-based checks; a flag means 'needs human review', not 'fraud'."}
