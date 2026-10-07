"""
selftest.py - "does it actually work?", answered on the machine itself.

The dependency report can read ok everywhere while the first render still
fails: weights that never finished downloading, a PyTorch that cannot see
the card, an engine that crashes on its first step. (The sibling studios
learned this the hard way; their Engine pages carry the same test.) So this
renders one short two-voice clip from MultiTalk's own example — the man and
the woman in examples/multi/1, taking turns — through the very path a real
render takes, then looks at what came back:

  1. the engine environment imports the engine
  2. PyTorch can see an NVIDIA card
  3. every weight file is on disk and the size it should be
  4. a 3-second two-voice clip renders                    (timed)
  5. the video has frames, they are not blank, and they move
  6. both voices are audible, each in its turn

It stops at the first step that breaks and says which, because "it does not
work" and "the text encoder stopped downloading at 2 GB" need different
fixes. Silence and blank frames count as failures: an MP4 of the right
length full of black decodes perfectly and shows nothing.

The media checks are pure functions over raw samples and pixels, so they are
unit-tested; ffmpeg does the decoding, from PATH or the copy imageio-ffmpeg
installed into the engine environment.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import bootstrap

FPS = 25
# The example's two lines: MultiTalk's own examples/multi/1 (a man, then a
# woman — "take turns"). Its picture has the two of them side by side.
EXAMPLE = ("examples", "multi", "1")
EXAMPLE_FILES = {"image": "multi1.png", "audio1": "1.WAV", "audio2": "2.WAV"}

STEPS = [("engine", "The engine environment imports the engine"),
         ("gpu", "PyTorch can see an NVIDIA card"),
         ("weights", "Every weight file is whole"),
         ("render", "A 3-second two-voice clip renders"),
         ("frames", "The video is 720 × 360, not blank, and it moves"),
         ("voices", "Both voices are audible, each in its turn")]


# --------------------------------------------------------------------------- #
# pure checks
# --------------------------------------------------------------------------- #
def weights_whole(wdir: Path, cfg: dict) -> tuple[bool, str]:
    """Every file present, and each big one at least half its published
    size — a cut-off download or a placeholder is present and useless."""
    missing, short = [], []
    for item in bootstrap.model_set(cfg):
        if item["role"] != "required":
            continue
        if not bootstrap.present(wdir, item):
            missing.append(item["name"])
            continue
        path = bootstrap.model_path(wdir, item)
        if not item["prefix"] and item["size"] >= 50_000_000:
            have = path.stat().st_size
            if have < item["size"] * 0.5:
                short.append(f"{item['name']} is {bootstrap.fmt_size(have)}, "
                             f"expected ≈ {bootstrap.fmt_size(item['size'])}")
    if missing:
        return False, "Missing: " + ", ".join(missing)
    if short:
        return False, ("Too small to be the real file — download it again "
                       "from the Models page: " + "; ".join(short))
    return True, "All required files present at their full size"


def frame_stats(frames: list[bytes]) -> dict:
    """Per-frame contrast and frame-to-frame change, from small grey frames."""
    def std(f: bytes) -> float:
        n = len(f) or 1
        mean = sum(f) / n
        return math.sqrt(sum((v - mean) ** 2 for v in f) / n)

    contrast = [std(f) for f in frames]
    moves = [sum(abs(a - b) for a, b in zip(f1, f2)) / max(len(f1), 1)
             for f1, f2 in zip(frames, frames[1:])]
    return {"frames": len(frames),
            "contrast": round(max(contrast), 2) if contrast else 0.0,
            "motion": round(max(moves), 2) if moves else 0.0}


def judge_frames(stats: dict, want_frames: int) -> tuple[bool, str]:
    if stats["frames"] < want_frames * 0.9:
        return False, (f"{stats['frames']} frames, expected about "
                       f"{want_frames}")
    if stats["contrast"] < 2:
        return False, ("Every frame is flat — a blank picture. The model ran "
                       "but drew nothing; the DiT weights are likely damaged.")
    if stats["motion"] < 0.2:
        return False, ("The frames never change — a still picture, not a "
                       "talking video.")
    return True, (f"{stats['frames']} frames · contrast {stats['contrast']:.0f} · "
                  f"they move")


def rms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def judge_voices(samples: list[int], rate: int, turns: list[float],
                 length: float) -> tuple[bool, str]:
    """`turns` are the seconds each voice speaks in order (take turns), and
    `length` is how long the clip is. Each turn that falls inside the clip
    must be audible in the middle of its own stretch."""
    if not samples:
        return False, "The video has no soundtrack."
    peak = max(abs(s) for s in samples) / 32768
    if peak < 0.01:
        return False, "The soundtrack is silent."
    levels, start = [], 0.0
    for i, dur in enumerate(turns):
        end = min(start + dur, length)
        if end - start < 0.3:
            break
        lo, hi = int((start + 0.1) * rate), int((end - 0.1) * rate)
        level = rms(samples[lo:hi]) / 32768
        levels.append(level)
        if level < 0.005:
            return False, (f"Voice {i + 1} is silent in its turn "
                           f"({start:.1f}–{end:.1f} s).")
        start = end
    if len(levels) < 2:
        return False, "The clip ended before the second voice's turn."
    return True, ("Voice 1 and voice 2 both audible · levels "
                  + " / ".join(f"{20 * math.log10(max(v, 1e-6)):.0f} dB"
                               for v in levels)
                  + f" · peak {peak * 100:.0f}%")


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, OSError, EOFError):
        return 0.0


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def find_ffmpeg(engine_python: str) -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    if engine_python:
        try:
            out = subprocess.run([engine_python, "-c", "import imageio_ffmpeg;"
                                  "print(imageio_ffmpeg.get_ffmpeg_exe())"],
                                 capture_output=True, text=True, timeout=60)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[-1]
        except Exception:  # noqa: BLE001
            pass
    return ""


def video_size(ffmpeg: str, video: Path) -> tuple[int, int]:
    """(width, height) of the first video stream, from ffmpeg's own banner."""
    import re
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)],
                         capture_output=True, text=True, timeout=60)
    m = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", out.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def read_frames(ffmpeg: str, video: Path, size: int = 32) -> list[bytes]:
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-vf",
                          f"scale={size}:{size},format=gray", "-f", "rawvideo",
                          "-"], capture_output=True, timeout=300)
    raw, n = out.stdout, size * size
    return [raw[i:i + n] for i in range(0, len(raw) - n + 1, n)]


def read_audio(ffmpeg: str, video: Path, rate: int = 8000) -> list[int]:
    out = subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-vn", "-ac",
                          "1", "-ar", str(rate), "-f", "s16le", "-"],
                         capture_output=True, timeout=300)
    raw = out.stdout[: len(out.stdout) // 2 * 2]
    return list(struct.unpack(f"<{len(raw) // 2}h", raw)) if raw else []


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class Run:
    """One self-test, step by step, for the page to poll."""

    def __init__(self) -> None:
        self.steps = [{"key": k, "label": v, "state": "pending", "detail": ""}
                      for k, v in STEPS]
        self.running = True
        self.ok = None
        self.job_id = ""
        self.clip = ""
        self.took = None
        self.estimate = ""

    def set(self, key: str, state: str, detail: str = "") -> None:
        for s in self.steps:
            if s["key"] == key:
                s["state"], s["detail"] = state, detail

    def fail(self, key: str, detail: str) -> None:
        self.set(key, "fail", detail)
        for s in self.steps:
            if s["state"] == "pending":
                s["state"] = "skipped"
        self.ok, self.running = False, False

    def view(self) -> dict:
        return {"steps": self.steps, "running": self.running, "ok": self.ok,
                "job": self.job_id, "clip": self.clip, "took": self.took,
                "estimate": self.estimate}


def estimate(took: float, clips_for_10s: int) -> str:
    """What a 10-second two-voice video costs on this machine, from the
    test's one clip (the model load is in both, so this is an upper bound)."""
    mins = took * clips_for_10s / 60
    return (f"≈ {mins:.0f} min for 10 s of speech on this machine "
            f"({clips_for_10s} clips; the first includes loading the model)")
