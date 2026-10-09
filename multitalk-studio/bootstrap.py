"""
bootstrap.py - first-launch setup for MultiTalk Studio.

MultiTalk is a 14B audio-driven video model (Wan2.1-I2V-14B plus MeiGen's
audio layers). In bf16 it wants ~30 GB of VRAM. This app runs the INT8 build
MeiGen publishes, keeps the DiT in system RAM and streams it layer by layer
(num_persistent_param_in_dit 0), encodes the prompt on the CPU and decodes the
video in tiles. That is what makes an 8 GB card such as an RTX 4060 usable.
preflight() measures the machine and says plainly what that means before
anything is downloaded.

Steps: Python -> engine environment -> PyTorch -> engine packages -> weights
-> check.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

# The engine never talks to localhost, but the page polls this server and a
# system proxy that does not exempt loopback turns every poll into a timeout.
_loopback = ["localhost", "127.0.0.1", "::1"]
for _key in ("NO_PROXY", "no_proxy"):
    _have = [h.strip() for h in os.environ.get(_key, "").split(",") if h.strip()]
    os.environ[_key] = ",".join(_have + [h for h in _loopback if h not in _have])

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
# Config, gallery, uploads and finished clips. MULTITALK_STUDIO_DATA moves the
# lot, which is what lets the tests run against a throwaway folder.
DATA_DIR = Path(os.environ.get("MULTITALK_STUDIO_DATA") or (APP_DIR / "data"))
CONFIG_PATH = DATA_DIR / "config.json"
ENGINE_REQUIREMENTS = APP_DIR / "engine-requirements.txt"

HF_BASE = "https://huggingface.co"
WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
MULTITALK_REPO = "MeiGen-AI/MeiGen-MultiTalk"
WAV2VEC_REPO = "TencentGameMate/chinese-wav2vec2-base"
KOKORO_REPO = "hexgrad/Kokoro-82M"

# Folder names under the weights directory — the same layout the upstream
# README uses, so a hand-made install is picked up as it is.
WAN_DIR = "Wan2.1-I2V-14B-480P"
MULTITALK_DIR = "MeiGen-MultiTalk"
WAV2VEC_DIR = "chinese-wav2vec2-base"
KOKORO_DIR = "Kokoro-82M"

# PyTorch is pinned to the build the upstream code was written against; the
# cu121 wheels cover every RTX 20/30/40 card, the RTX 4060 included.
TORCH_SPEC = ["torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1"]
XFORMERS_SPEC = "xformers==0.0.28.post1"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"

# Sizes are what the app shows before HuggingFace is asked. The quantised
# files are approximate (marked so in the UI); the setup run asks the repo
# for the real numbers before it starts.
PRECISIONS = {
    "int8-fusionx": {
        "label": "INT8 + FusionX — 8 steps (recommended for 8 GB)",
        "note": "MeiGen's INT8 model with the FusionX acceleration merged in. "
                "8 sampling steps instead of 40, so a clip takes minutes, not "
                "most of an hour, on an RTX 4060.",
        "dit": {"path": "quant_models/quant_model_int8_FusionX.safetensors",
                "size": 16_500_000_000, "approx": True},
        "map": {"path": "quant_models/quantization_map_int8_FusionX.json",
                "size": 300_000, "approx": True},
        "defaults": {"steps": 8, "text_scale": 1.0, "audio_scale": 2.0,
                     "shift": 2.0, "teacache": False},
    },
    "int8": {
        "label": "INT8 — base model, 40 steps",
        "note": "MeiGen's INT8 model as trained. Best prompt following, about "
                "five times slower than FusionX. TeaCache claws back about 2x.",
        "dit": {"path": "quant_models/dit_model_int8.safetensors",
                "size": 16_500_000_000, "approx": True},
        "map": {"path": "quant_models/dit_model_map_int8.json",
                "size": 300_000, "approx": True},
        "defaults": {"steps": 40, "text_scale": 5.0, "audio_scale": 4.0,
                     "shift": 7.0, "teacache": True},
    },
}

T5_INT8 = {"path": "quant_models/t5_int8.safetensors", "size": 6_700_000_000,
           "approx": True}
T5_MAP = {"path": "quant_models/t5_map_int8.json", "size": 100_000,
          "approx": True}

SIZES = {
    "multitalk-240": {"label": "320 px", "px": 320,
                      "note": "Fastest. A draft size; faces are soft."},
    "multitalk-360": {"label": "480 px", "px": 480,
                      "note": "The 8 GB sweet spot."},
    "multitalk-480": {"label": "640 px", "px": 640,
                      "note": "The size MultiTalk was trained at. Needs 12 GB+ "
                              "to be comfortable."},
}

DEFAULT_CONFIG = {
    "engine_dir": str(REPO_DIR / "MultiTalk"),
    "weights_dir": "",            # empty: <engine_dir>/weights
    "python": "",                 # the engine's interpreter, set by setup
    "torch_index": "",
    "hf_token": "", "hf_endpoint": HF_BASE,
    "precision": "int8-fusionx",
    "want_tts": True,
    "want_xformers": True,
    "setup_complete": False,
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
_write_lock = threading.Lock()


def atomic_write(path: Path, text: str) -> None:
    """Write through a temporary file and swap it in, so a crash mid-write
    never leaves half a JSON document. On Windows the swap is retried: an
    antivirus scan or OneDrive sync briefly holding the file is routine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.1)


def quarantine(path: Path) -> Path | None:
    """Move a file that will not parse aside, so the next save cannot bury
    what was in it."""
    dest = path.with_name(f"{path.name}.bad-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        os.replace(path, dest)
        print(f"[multitalk-studio] {path.name} could not be read; kept as "
              f"{dest.name}", flush=True)
        return dest
    except OSError:
        return None


def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("not an object")
            cfg.update(saved)
        except (OSError, ValueError):
            quarantine(CONFIG_PATH)
    if cfg.get("precision") not in PRECISIONS:
        cfg["precision"] = DEFAULT_CONFIG["precision"]
    return cfg


def save_config(cfg: dict) -> None:
    with _write_lock:
        atomic_write(CONFIG_PATH, json.dumps(cfg, indent=2))


def engine_dir(cfg: dict) -> Path:
    return Path(cfg.get("engine_dir") or DEFAULT_CONFIG["engine_dir"])


def weights_dir(cfg: dict) -> Path:
    if cfg.get("weights_dir"):
        return Path(cfg["weights_dir"])
    return engine_dir(cfg) / "weights"


# --------------------------------------------------------------------------- #
# model set
# --------------------------------------------------------------------------- #
def model_set(cfg: dict) -> list[dict]:
    """Every file the chosen precision needs, with where it goes.

    An item with "prefix" stands for a whole folder of the repo (the two
    tokenizers and the Kokoro voices): it is listed at download time and
    counts as present once its tokenizer assets (or voice files) are present.
    """
    p = PRECISIONS.get(cfg.get("precision") or "", PRECISIONS["int8-fusionx"])

    def item(repo, root, path, size, why, role="required", prefix=False,
             revision="main", approx=False, group=""):
        return {"repo": repo, "root": root, "path": path, "size": size,
                "why": why, "role": role, "prefix": prefix,
                "revision": revision, "approx": approx, "group": group,
                "name": path.rstrip("/").split("/")[-1] + ("/" if prefix else "")}

    items = [
        item(MULTITALK_REPO, MULTITALK_DIR, p["dit"]["path"], p["dit"]["size"],
             "The MultiTalk video model, quantised to INT8 — audio and a "
             "picture in, talking video out.", approx=True, group="MultiTalk"),
        item(MULTITALK_REPO, MULTITALK_DIR, p["map"]["path"], p["map"]["size"],
             "How the INT8 weights unpack.", approx=True, group="MultiTalk"),
        item(MULTITALK_REPO, MULTITALK_DIR, T5_INT8["path"], T5_INT8["size"],
             "The umT5-XXL text encoder, INT8. Reads the prompt.",
             approx=True, group="MultiTalk"),
        item(MULTITALK_REPO, MULTITALK_DIR, T5_MAP["path"], T5_MAP["size"],
             "How the INT8 text encoder unpacks.", approx=True,
             group="MultiTalk"),
        item(WAN_REPO, WAN_DIR, "config.json", 1_000,
             "The model's shape.", group="Wan2.1 base"),
        item(WAN_REPO, WAN_DIR, "Wan2.1_VAE.pth", 507_609_880,
             "The video VAE — encodes the reference picture, decodes the "
             "video.", group="Wan2.1 base"),
        item(WAN_REPO, WAN_DIR,
             "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
             4_772_669_925, "The CLIP image encoder that reads the reference "
             "picture.", group="Wan2.1 base"),
        item(WAN_REPO, WAN_DIR, "google/umt5-xxl/", 21_000_000,
             "The text encoder's tokenizer.", prefix=True, approx=True,
             group="Wan2.1 base"),
        item(WAN_REPO, WAN_DIR, "xlm-roberta-large/", 22_000_000,
             "The image encoder's tokenizer.", prefix=True, approx=True,
             group="Wan2.1 base"),
        item(WAV2VEC_REPO, WAV2VEC_DIR, "config.json", 2_000,
             "The audio encoder's shape.", group="Audio encoder"),
        item(WAV2VEC_REPO, WAV2VEC_DIR, "preprocessor_config.json", 1_000,
             "How audio is fed to the encoder.", group="Audio encoder"),
        # the safetensors build lives on a PR branch, as the upstream README
        # downloads it
        item(WAV2VEC_REPO, WAV2VEC_DIR, "model.safetensors", 380_000_000,
             "wav2vec2 — turns speech into what drives the lips.",
             revision="refs/pr/1", approx=True, group="Audio encoder"),
    ]
    if cfg.get("want_tts", True):
        items += [
            item(KOKORO_REPO, KOKORO_DIR, "config.json", 2_000,
                 "Kokoro's shape.", role="optional", group="Text to speech"),
            item(KOKORO_REPO, KOKORO_DIR, "kokoro-v1_0.pth", 327_212_226,
                 "Kokoro-82M — speaks typed text, so a clip needs no "
                 "recording.", role="optional", approx=True,
                 group="Text to speech"),
            item(KOKORO_REPO, KOKORO_DIR, "voices/", 28_000_000,
                 "The voices to choose from.", role="optional", prefix=True,
                 approx=True, group="Text to speech"),
        ]
    return items


def model_path(wdir: Path, item: dict) -> Path:
    return wdir / item["root"] / item["path"]


def finished_file(path: Path, expected_size: int = 0) -> bool:
    """Reject empty files, partial downloads and Git LFS pointer stubs.

    This is a quick presence check, not a checksum or a model-load test.
    An exact byte count is enforced only when supplied by a remote file list.
    """
    try:
        if not path.is_file() or path.suffix in (".part", ".incomplete"):
            return False
        if path.resolve().suffix in (".part", ".incomplete"):
            return False
        size = path.stat().st_size
        if size <= 0 or (expected_size and size != expected_size):
            return False
        with path.open("rb") as fh:
            return not fh.read(64).startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def folder_present(target: Path, prefix: str) -> bool:
    """The assets AutoTokenizer needs, rather than an arbitrary README."""
    if prefix in ("google/umt5-xxl/", "xlm-roberta-large/"):
        vocab = ("tokenizer.json", "spiece.model", "sentencepiece.bpe.model")
        return (finished_file(target / "tokenizer_config.json")
                and any(finished_file(target / name) for name in vocab))
    if prefix == "voices/":
        return any(finished_file(f) for f in target.glob("*.pt"))
    try:
        return any(finished_file(f) for f in target.rglob("*"))
    except OSError:
        return False


def present(wdir: Path, item: dict) -> bool:
    target = model_path(wdir, item)
    if item.get("prefix"):
        return folder_present(target, item["path"])
    return finished_file(target)


def missing_models(wdir: Path, cfg: dict, required_only: bool = True) -> list[dict]:
    return [m for m in model_set(cfg)
            if (m["role"] == "required" or not required_only)
            and not present(wdir, m)]


# --------------------------------------------------------------------------- #
# human-readable numbers
# --------------------------------------------------------------------------- #
def fmt_size(n: float) -> str:
    n = float(n or 0)
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} kB"
    return f"{int(n)} B"


def fmt_eta(seconds: float) -> str:
    s = int(max(seconds or 0, 0))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m left"
    if s >= 60:
        return f"{s // 60}m {s % 60}s left"
    return f"{s}s left"


def fmt_transfer(got: float, total: float, speed: float, eta: float) -> str:
    bits = [f"{fmt_size(got)} of {fmt_size(total)}" if total
            else f"{fmt_size(got)} so far"]
    if speed > 0:
        bits.append(f"{speed / 1e6:.1f} MB/s")
    if total and speed > 0:
        bits.append(fmt_eta(eta))
    return " · ".join(bits)


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
GIB = 1024 ** 3


def _ram_bytes() -> int:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MS()
            stat.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            return 0
    return 0


_GPU_SEEN: dict[str, tuple[int, str]] = {}


def gpu_info(python: str, fresh: bool = False) -> tuple[int, str]:
    """(VRAM bytes, GPU name) as the engine's own torch sees it; cached,
    since importing torch takes seconds and the answer only changes when
    torch is reinstalled."""
    if not python or not Path(python).exists():
        return 0, ""
    if python in _GPU_SEEN and not fresh:
        return _GPU_SEEN[python]
    code = ("import torch,json;d=torch.cuda.is_available();"
            "print(json.dumps({'v':(torch.cuda.get_device_properties(0)"
            ".total_memory if d else 0),"
            "'n':(torch.cuda.get_device_name(0) if d else '')}))")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True,
                             text=True, timeout=120)
        if out.returncode != 0:
            return 0, ""
        d = json.loads(out.stdout.strip().splitlines()[-1])
        result = (int(d["v"]), d["n"])
        _GPU_SEEN[python] = result
        return result
    except Exception:
        return 0, ""


def forget_gpu() -> None:
    _GPU_SEEN.clear()


def peak_ram(cfg: dict) -> int:
    """System RAM a render holds at its peak: the INT8 DiT and text encoder
    both live in RAM (the DiT streams to the GPU a layer at a time, the text
    encoder runs on the CPU), plus CLIP while it loads and about 3 GB of
    Python, wav2vec and buffers."""
    items = {m["path"]: m["size"] for m in model_set(cfg)}
    p = PRECISIONS.get(cfg.get("precision") or "", PRECISIONS["int8-fusionx"])
    return (items.get(p["dit"]["path"], 0) + T5_INT8["size"]
            + 4_772_669_925 + 3_000_000_000)


def assess(vram: int, ram: int, free_disk: int, download: int,
           peak: int) -> tuple[str, list[str]]:
    """The verdict from the measurements.

    The VRAM floor is an estimate from what a render holds on the GPU with
    this app's 8 GB settings: the VAE (~0.3 GB), one DiT block at a time
    (~0.4 GB in INT8) and the activations of a 480 px, 81-frame clip
    (~3-4 GB). CLIP (~2.4 GB) visits the GPU before sampling and is sent
    back to RAM; the text encoder never comes. These estimates have not
    been validated on a real 8 GB card; that is "tight", not "hard". "Hard"
    is kept for what genuinely blocks: under ~6 GB of VRAM, RAM too small
    to hold the weights, disk short of the download.
    """
    notes, verdict = [], "ok"

    def worse(level: str) -> None:
        nonlocal verdict
        order = ("ok", "tight", "hard")
        if order.index(level) > order.index(verdict):
            verdict = level

    if vram and vram < 5.5 * GIB:
        worse("hard")
        notes.append(f"{vram / GIB:.0f} GB of VRAM is under the ~6 GB the "
                     "smallest size needs even with everything offloaded. "
                     "Expect out-of-memory stops.")
    elif vram and vram < 10 * GIB:
        worse("tight")
        notes.append(f"{vram / GIB:.0f} GB of VRAM — try this app's "
                     "low-memory settings: INT8 weights streamed from RAM, the "
                     "prompt encoded on the CPU, tiled VAE, 480 px. Close "
                     "games and browsers with hardware acceleration while it "
                     "renders. Real-checkpoint 8 GB operation is not yet verified.")
    elif vram and vram < 16 * GIB:
        notes.append(f"{vram / GIB:.0f} GB of VRAM — 640 px, the trained "
                     "size, is within reach.")
    if ram and peak and ram < peak * 0.8:
        worse("hard")
        notes.append(f"{ram / GIB:.0f} GB of system RAM against a "
                     f"~{peak / GIB:.0f} GB peak. The engine maps the INT8 "
                     "model from its file, but activations and outputs still "
                     "need RAM. It may stop or page heavily: keep "
                     "the weights on an SSD (NVMe if you have one), never a "
                     "hard drive, and expect renders many times slower. "
                     "32 GB is a starting point, not a fit guarantee.")
    elif ram and peak and ram < peak * 1.1:
        worse("tight")
        notes.append(f"{ram / GIB:.0f} GB of system RAM against a "
                     f"~{peak / GIB:.0f} GB estimated peak — little headroom. Close "
                     "other programs before a render, and keep the page file "
                     "on an SSD.")
    if free_disk and download and free_disk < download * 1.1:
        worse("hard")
        notes.append(f"{free_disk / 1e9:.0f} GB free where the weights go; "
                     f"the set needs ~{download / 1e9:.0f} GB.")
    if not vram:
        notes.append("Could not read an NVIDIA GPU. Before setup that is "
                     "expected — PyTorch is not installed yet. After setup it "
                     "means PyTorch cannot see the card: update the NVIDIA "
                     "driver, then press Measure again.")
    if verdict == "hard":
        notes.append("It will still install and queue; generation may fail "
                     "or be too slow to use.")
    return verdict, notes


def recommended(vram: int) -> dict:
    """Settings that fit the card. The page applies these as its defaults."""
    if vram and vram >= 20 * GIB:
        return {"size": "multitalk-480", "t5_cpu": False, "vae_tile": 0}
    if vram and vram >= 12 * GIB:
        return {"size": "multitalk-480", "t5_cpu": True, "vae_tile": 0}
    return {"size": "multitalk-360", "t5_cpu": True, "vae_tile": 32}


def preflight(cfg: dict) -> dict:
    vram, gpu = gpu_info(engine_python(cfg))
    ram = _ram_bytes()
    wdir = weights_dir(cfg)
    probe = wdir
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free_disk = shutil.disk_usage(str(probe)).free
    except Exception:
        free_disk = 0
    items = model_set(cfg)
    download = sum(i["size"] for i in items)
    still = sum(i["size"] for i in items if not present(wdir, i))
    peak = peak_ram(cfg)
    verdict, notes = assess(vram, ram, free_disk, still or download, peak)
    return {"vram": vram, "gpu": gpu, "ram": ram, "free_disk": free_disk,
            "download": download, "to_download": still, "peak": peak,
            "verdict": verdict, "notes": notes,
            "precision": cfg.get("precision"),
            "recommended": recommended(vram)}


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
class Progress:
    STEPS = [("python", "Check Python"),
             ("venv", "Create the engine environment"),
             ("torch", "Install PyTorch"),
             ("packages", "Install the engine packages"),
             ("models", "Download the weights"),
             ("check", "Check the engine loads")]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []
        self.running = False
        self.done = False
        self.error: str | None = None
        self.step = ""
        self.steps = {k: {"key": k, "label": v, "state": "pending",
                          "detail": "", "pct": None} for k, v in self.STEPS}

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 4000:
                del self.lines[:2000]
        print(f"[setup] {msg}", flush=True)

    def begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.step = key
            self.steps[key].update(state="running", detail=detail, pct=None)

    def track(self, key: str, pct: float | None, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["pct"] = (None if pct is None
                                      else round(max(0.0, min(100.0, pct)), 1))
            if detail:
                self.steps[key]["detail"] = detail

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key].update(state="done", pct=None)
            if detail:
                self.steps[key]["detail"] = detail

    def fail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key].update(state="error", pct=None, detail=detail)

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            steps = [dict(self.steps[k]) for k, _ in self.STEPS]
            return {"running": self.running, "done": self.done,
                    "error": self.error, "step": self.step, "steps": steps,
                    "cursor": len(self.lines), "lines": self.lines[since:]}


# --------------------------------------------------------------------------- #
# interpreters
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def stream(cmd: list[str], on_line, cwd: str | None = None,
           env: dict | None = None, should_cancel=None,
           on_start=None) -> int:
    r"""Run `cmd` and hand every line of its output to `on_line` as it appears.

    Splits on carriage returns as well as newlines: tqdm and pip redraw one
    line with \r, so a plain line iterator would hold all of it back until
    the end — exactly the silence this is meant to fill.
    """
    kw = {}
    if platform.system() == "Windows":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=cwd, env=env, **kw)
    if on_start:
        on_start(proc)
    assert proc.stdout
    buf = b""
    while True:
        block = proc.stdout.read1(8192)
        if not block:
            break
        if should_cancel and should_cancel():
            kill_tree(proc)
            return proc.wait()
        buf += block
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            text = raw.decode("utf-8", "replace").strip()
            if text:
                on_line(text)
    if buf.strip():
        on_line(buf.decode("utf-8", "replace").strip())
    return proc.wait()


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop a process and everything it started (ffmpeg included)."""
    if proc.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=30)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


# torch 2.4.1 publishes wheels for Python 3.8-3.12; 3.13 has none
ENGINE_PY_MIN, ENGINE_PY_MAX = (3, 10), (3, 12)


def find_python(prog: Progress | None = None) -> str:
    """A Python the engine can live on, tested by running it — Windows Store
    stubs answer on PATH and fail to execute."""
    candidates: list[list[str]] = [[sys.executable]]
    if platform.system() == "Windows":
        candidates += [["py", "-3.12"], ["py", "-3.11"], ["py", "-3.10"],
                       ["py", "-3"], ["python"]]
    else:
        candidates += [["python3.12"], ["python3.11"], ["python3.10"],
                       ["python3"], ["python"]]
    seen = []
    for cand in candidates:
        try:
            out = _run(cand + ["-c", "import sys;print(sys.executable);"
                                     "print('%d.%d' % sys.version_info[:2])"],
                       timeout=25)
        except Exception:
            continue
        if out.returncode != 0:
            continue
        parts = [p.strip() for p in out.stdout.strip().splitlines() if p.strip()]
        if len(parts) < 2:
            continue
        try:
            ver = tuple(int(x) for x in parts[1].split("."))
        except ValueError:
            continue
        seen.append(parts[1])
        if ENGINE_PY_MIN <= ver <= ENGINE_PY_MAX:
            if prog:
                prog.log(f"Using Python {parts[1]} at {parts[0]}")
            return parts[0]
    found = (" Found " + ", ".join(sorted(set(seen))) + ".") if seen else ""
    raise RuntimeError("The engine needs Python 3.10, 3.11 or 3.12 (PyTorch "
                       "2.4 has no wheels for newer ones)." + found +
                       " Install 3.12 from python.org, then run setup again.")


def venv_dir() -> Path:
    return APP_DIR / "engine-venv"


def venv_python() -> Path:
    return venv_dir() / ("Scripts/python.exe" if platform.system() == "Windows"
                         else "bin/python")


def engine_python(cfg: dict) -> str:
    if cfg.get("python") and Path(cfg["python"]).exists():
        return cfg["python"]
    v = venv_python()
    return str(v) if v.exists() else ""


def torch_index(cfg: dict) -> str:
    if cfg.get("torch_index"):
        return cfg["torch_index"]
    if platform.system() == "Darwin":
        return ""
    return DEFAULT_TORCH_INDEX


# --------------------------------------------------------------------------- #
# saved locations
# --------------------------------------------------------------------------- #
def rebase_path(old: str) -> Path | None:
    """Where a path saved under an earlier location of this repo lives now.

    The config keeps absolute paths — the default engine folder included.
    Move, rename or re-extract the repo (Multi-Talk -> Multi-Talk-main,
    C: -> D:) and every one of them points nowhere, though the engine, its
    weights and the engine environment moved with it. The tail of the old
    path is the same; graft it onto this repo or this app, trying the
    longest tail first, so a renamed repo folder is no obstacle.
    """
    parts = [p for p in re.split(r"[\\/]+", old or "") if p]
    for i in range(1, len(parts)):
        tail = parts[i:]
        if ".." in tail:
            continue
        for base in (REPO_DIR, APP_DIR):
            cand = base.joinpath(*tail)
            try:
                if cand.exists():
                    return cand
            except OSError:
                continue
    return None


def has_engine(path: Path | None) -> bool:
    """The test the rest of the app uses: the engine's entry script is there."""
    try:
        return bool(path) and (path / "generate_multitalk.py").is_file()
    except OSError:
        return False


def heal_paths(cfg: dict) -> list[str]:
    """Repair saved paths that no longer exist. Returns what changed."""
    notes: list[str] = []
    old_eng = cfg.get("engine_dir") or ""
    eng = engine_dir(cfg)
    if not has_engine(eng):
        for c in (rebase_path(old_eng), Path(DEFAULT_CONFIG["engine_dir"])):
            if has_engine(c) and str(c) != old_eng:
                cfg["engine_dir"] = str(c)
                eng = c
                notes.append(f"MultiTalk engine found at {c}")
                break
    old_w = cfg.get("weights_dir") or ""
    if old_w and not Path(old_w).is_dir():
        moved = rebase_path(old_w)
        if moved and moved.is_dir():
            cfg["weights_dir"] = str(moved)
            notes.append(f"Weights folder found at {moved}")
        elif (eng / "weights").is_dir():
            cfg["weights_dir"] = ""     # the default: <engine_dir>/weights
            notes.append(f"Weights folder found at {eng / 'weights'}")
    py = cfg.get("python") or ""
    if py and not Path(py).exists():
        moved = rebase_path(py)
        moved = moved if moved and moved.is_file() else None
        cfg["python"] = str(moved) if moved else ""
        notes.append(f"Engine Python {py} is gone"
                     + (f"; using {moved}" if moved else "; cleared"))
    return notes


# folders never worth walking into when hunting for the engine: system trees,
# package caches, environments, and weight folders (thousands of entries)
_SKIP_DIRS = {"windows", "program files", "program files (x86)", "programdata",
              "appdata", "$recycle.bin", "system volume information",
              "recovery", "node_modules", ".git", "__pycache__",
              "site-packages", "lib", "libs", "scripts", ".cache", ".venv",
              "venv", "engine-venv", "weights", "proc", "sys", "dev", "snap"}


def is_engine_dir(path: Path) -> bool:
    """A MultiTalk checkout: the entry script and the pipeline it imports
    (`wan/multitalk.py`, the `wan.MultiTalkPipeline` the script runs)."""
    try:
        return (path / "generate_multitalk.py").is_file() and \
            (path / "wan" / "multitalk.py").is_file()
    except OSError:
        return False


def search_roots() -> list[Path]:
    """Where a full search starts: around the repo, home, then every drive."""
    roots = [REPO_DIR.parent, Path.home()]
    if platform.system() == "Windows":
        roots += [Path(f"{c}:/") for c in "CDEFGHIJKLMNOPQRSTUVWXYZ"
                  if os.path.exists(f"{c}:/")]
    else:
        roots += [Path("/opt"), Path("/srv"), Path("/mnt"), Path("/media")]
    return roots


def find_engine_installs(roots: list[Path] | None = None, max_depth: int = 6,
                         budget: float = 45.0) -> list[Path]:
    """Every MultiTalk checkout under `roots`, shallowest first, within a
    time budget.

    Breadth-first, so the copy a person put somewhere sensible is met long
    before the walk wanders into deep trees, and a slow or huge drive ends
    the search on time.
    """
    deadline = time.monotonic() + budget
    found: list[Path] = []
    seen: set[str] = set()
    queue = [(r, 0) for r in (roots if roots is not None else search_roots())]
    while queue and time.monotonic() < deadline:
        path, depth = queue.pop(0)
        try:
            key = os.path.normcase(str(path.resolve()))
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        if is_engine_dir(path):
            found.append(path)
            continue                # nothing worth finding inside one
        if depth >= max_depth:
            continue
        try:
            with os.scandir(path) as it:
                for e in it:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        if not e.is_dir(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    if e.name.lower() in _SKIP_DIRS or e.name.startswith("."):
                        continue
                    queue.append((Path(e.path), depth + 1))
        except OSError:
            continue
    return found


def pick_engine(installs: list[Path], cfg: dict) -> Path | None:
    """The checkout to use: the one holding the weights, then the one inside
    this repo, then the first found."""
    def score(c: Path) -> tuple:
        w = c / "weights"
        weights = w.is_dir() and not missing_models(w, cfg)
        try:
            inside = c.resolve().is_relative_to(REPO_DIR.resolve())
        except (OSError, ValueError):
            inside = False
        return (not weights, not inside)
    return min(installs, key=score) if installs else None


def location_key(cfg: dict) -> list:
    """Cheap path health only; no tree or drive traversal."""
    py = cfg.get("python") or ""
    return [str(REPO_DIR), str(engine_dir(cfg)), has_engine(engine_dir(cfg)),
            str(weights_dir(cfg)), weights_dir(cfg).is_dir(), py,
            bool(py and Path(py).is_file())]


def needs_location_search(cfg: dict) -> bool:
    return (not has_engine(engine_dir(cfg))
            and cfg.get("_location_search_attempt") != location_key(cfg))


def verify_locations(cfg: dict, search: bool = True, log=None,
                     force: bool = False) -> list[str]:
    """Reuse verified locations, with one recovery attempt per failed state.

    Both successful checks and unsuccessful searches are persisted in config.
    A path change/failure permits a fresh attempt; only an explicit Recheck
    bypasses the remembered result when the same paths remain unavailable.
    """
    say = log or (lambda _m: None)
    notes = []
    key = location_key(cfg)
    if force or cfg.get("_location_check") != key:
        notes = heal_paths(cfg)
        cfg["_location_check"] = location_key(cfg)
    if has_engine(engine_dir(cfg)):
        cfg.pop("_location_search_attempt", None)
        return notes
    if not search or (not force and not needs_location_search(cfg)):
        return notes
    cfg["_location_search_attempt"] = location_key(cfg)
    say("Searching this computer for the MultiTalk engine…")
    hit = pick_engine(find_engine_installs(), cfg)
    if not hit:
        say("No MultiTalk engine found on this computer — set its folder "
            "in Settings or use Recheck to search again.")
        return notes
    cfg["engine_dir"] = str(hit)
    notes.append(f"MultiTalk engine found at {hit}")
    w = cfg.get("weights_dir") or ""
    if w and not Path(w).is_dir() and (hit / "weights").is_dir():
        cfg["weights_dir"] = ""
        notes.append(f"Weights folder found at {hit / 'weights'}")
    cfg["_location_check"] = location_key(cfg)
    cfg.pop("_location_search_attempt", None)
    return notes


def location_report(cfg: dict) -> list[str]:
    """One line per saved location, saying whether it checks out."""
    eng = engine_dir(cfg)
    out = [f"Engine: {eng} — ok" if has_engine(eng) else "Engine: not found"]
    w = weights_dir(cfg)
    if w.is_dir():
        gone = missing_models(w, cfg)
        out.append(f"Weights: {w} — " + ("all required files present"
                                          if not gone else
                                          f"{len(gone)} required file(s) missing"))
    else:
        out.append("Weights: not found")
    py = engine_python(cfg)
    out.append(f"Engine Python: {py} — ok" if py
               else "Engine Python: not set up yet")
    return out


# --------------------------------------------------------------------------- #
# pip
# --------------------------------------------------------------------------- #
PIP_RAW = re.compile(r"^Progress (\d+) of (\d+)$")
PIP_GET = re.compile(r"^\s*(?:Downloading|Using cached)\s+(\S+)")
_PIP_RAW_OK: dict[str, bool] = {}


def pip_has_raw_progress(python: str) -> bool:
    """Whether this pip takes --progress-bar raw (pip 24.1 and newer).

    Asked by letting pip parse the flag: an older pip rejects the choice
    and exits 2, a newer one prints its help and exits 0. Reading the help
    text instead broke on how the console wraps it."""
    if python not in _PIP_RAW_OK:
        try:
            out = _run([python, "-m", "pip", "install", "--progress-bar", "raw",
                        "--help"], timeout=60)
            ok = out.returncode == 0
        except Exception:  # noqa: BLE001
            ok = False
        _PIP_RAW_OK[python] = ok
    return _PIP_RAW_OK[python]


def pip_install(python: str, args: list[str], log, on_pct=None,
                should_cancel=None) -> None:
    """Install with pip; `on_pct(pct|None, detail)` follows the download.
    pip's own bar vanishes in a pipe, so a 2.4 GB torch wheel would look
    like a hang without --progress-bar raw."""
    import urllib.parse
    cmd = [python, "-m", "pip", "install", "--disable-pip-version-check"]
    if on_pct and pip_has_raw_progress(python):
        cmd += ["--progress-bar", "raw"]
    cmd += args
    log("$ " + " ".join(cmd[:9]) + (" …" if len(cmd) > 9 else ""))
    cur = {"name": "", "base": 0, "started": 0.0, "last": 0.0}

    def line(text: str) -> None:
        m = PIP_RAW.match(text)
        if m:
            if not on_pct:
                return
            got, total = int(m.group(1)), int(m.group(2))
            now = time.time()
            if got < cur["base"] or not cur["started"]:
                cur["base"], cur["started"] = got, now
            # pip already limits raw progress to four lines a second; every
            # one is passed on, so a fast wheel still shows its middle
            cur["last"] = now
            speed = (got - cur["base"]) / max(now - cur["started"], .1)
            eta = (total - got) / speed if speed > 0 and total else 0
            on_pct((got / total * 100) if total else None,
                   f"{cur['name'] or 'package'} — "
                   f"{fmt_transfer(got, total, speed, eta)}")
            return
        m = PIP_GET.match(text)
        if m:
            cur.update(name=urllib.parse.unquote(
                m.group(1).rsplit("/", 1)[-1])[:60],
                base=0, started=0.0, last=0.0)
        if text.startswith(("Collecting", "Downloading", "Installing",
                            "Successfully", "ERROR", "Building", "WARNING: ")):
            log(text[:200])
            if on_pct and text.startswith(("Installing", "Building")):
                on_pct(None, text[:120])

    if stream(cmd, line, should_cancel=should_cancel) != 0:
        if should_cancel and should_cancel():
            raise RuntimeError("Cancelled.")
        raise RuntimeError("pip install failed — see the log.")
    if "pip" in args:
        _PIP_RAW_OK.pop(python, None)


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
def hf_headers(cfg: dict) -> dict:
    token = (cfg.get("hf_token") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_endpoint(cfg: dict) -> str:
    return (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")


def _rev(revision: str) -> str:
    import urllib.parse
    return urllib.parse.quote(revision or "main", safe="")


def hf_tree(cfg: dict, repo: str, revision: str = "main") -> list[dict]:
    base = hf_endpoint(cfg)
    url = f"{base}/api/models/{repo}/tree/{_rev(revision)}?recursive=1"
    try:
        r = requests.get(url, headers=hf_headers(cfg), timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not reach {base}: {exc}") from exc
    if r.status_code == 401:
        raise RuntimeError("This repo needs a HuggingFace token. Add one on "
                           "the Models page, then try again.")
    if r.status_code == 403:
        raise RuntimeError("Your token cannot read this repo — accept its "
                           "terms on the model page first.")
    if r.status_code == 404:
        raise RuntimeError(f"Could not find '{repo}' on {base}.")
    r.raise_for_status()
    files = []
    for e in r.json():
        if e.get("type") != "file":
            continue
        size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
        files.append({"path": e["path"], "size": size})
    return files


def hf_cache_roots() -> list[Path]:
    """Use the same cache locations as huggingface_hub, without importing it."""
    home = Path(os.environ.get("HF_HOME") or
                Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                / "huggingface").expanduser()
    roots = [Path(os.environ[key]).expanduser() for key in
             ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
             if os.environ.get(key)]
    roots.append(home / "hub")
    return list(dict.fromkeys(roots))


def cached_snapshots(repo: str, revision: str = "main") -> list[Path]:
    """Only the requested repo and ref; never guess another model/revision."""
    if any(p in ("", ".", "..") for p in repo.split("/")):
        return []
    if any(p in ("", ".", "..") for p in revision.split("/")):
        return []
    out = []
    for root in hf_cache_roots():
        base = root / ("models--" + repo.replace("/", "--"))
        try:
            commit = revision
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                commit = (base / "refs" / revision).read_text().strip()
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                continue
            snapshot = base / "snapshots" / commit
            if snapshot.is_dir():
                out.append(snapshot)
        except (OSError, UnicodeError):
            continue
    return out


def cached_file(repo: str, path: str, revision: str = "main",
                expected_size: int = 0) -> Path | None:
    if Path(path).is_absolute() or "\\" in path or ".." in path.split("/"):
        return None
    for snapshot in cached_snapshots(repo, revision):
        source = snapshot / path
        if finished_file(source, expected_size):
            if not expected_size:
                estimates = [m["size"] for precision in PRECISIONS
                             for m in model_set({"precision": precision})
                             if m["repo"] == repo and m["path"] == path]
                if estimates and max(estimates) >= 1_000_000 and \
                        source.stat().st_size < max(estimates) // 2:
                    continue
            return source
    return None


def cached_items(item: dict) -> list[dict] | None:
    """Expand a usable cached item without a Hub request.

    A snapshot can itself be partial: tokenizer folders must have their config
    and vocabulary. Large weights below half the published estimate are not
    adopted as complete. These quick checks do not validate model contents.
    """
    for snapshot in cached_snapshots(item["repo"], item.get("revision", "main")):
        source = snapshot / item["path"]
        if item.get("prefix"):
            if not folder_present(source, item["path"]):
                continue
            files = [f for f in source.rglob("*") if finished_file(f)]
        elif finished_file(source):
            estimate = item.get("size", 0)
            if estimate >= 1_000_000 and source.stat().st_size < estimate // 2:
                continue
            files = [source]
        else:
            continue
        return [{**item, "path": f.relative_to(snapshot).as_posix(),
                 "name": f.name, "prefix": False, "size": f.stat().st_size,
                 "approx": False, "known_size": f.stat().st_size}
                for f in files]
    return None


def reuse_cache(source: Path, dest: Path, on_progress=None,
                should_cancel=None) -> bool:
    """Adopt immutable cached bytes; retain the original and any .part file.

    A hard link saves space on the same drive. Different drives use an atomic,
    cancellable copy. Neither path opens the cache file for writing.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + f".{threading.get_ident()}.reuse")
    size = source.stat().st_size
    try:
        if should_cancel and should_cancel():
            return False
        try:
            os.link(source.resolve(), temp)
        except OSError:
            got, started = 0, time.monotonic()
            with source.open("rb") as src, temp.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    if should_cancel and should_cancel():
                        return False
                    dst.write(chunk)
                    got += len(chunk)
                    if on_progress:
                        speed = got / max(time.monotonic() - started, .001)
                        on_progress(got, size, speed, (size - got) / speed)
        if should_cancel and should_cancel():
            return False
        temp.replace(dest)
        if on_progress:
            on_progress(size, size, 0, 0)
        return True
    finally:
        temp.unlink(missing_ok=True)


# one writer per .part — setup and the Models page could both append
_writing: set[Path] = set()
_writing_lock = threading.Lock()


def download_file(cfg: dict, repo: str, path: str, dest: Path,
                  on_progress=None, should_cancel=None,
                  revision: str = "main", expected_size: int = 0) -> None:
    with _writing_lock:
        if dest in _writing:
            raise RuntimeError(f"{dest.name} is already downloading.")
        _writing.add(dest)
    try:
        if finished_file(dest, expected_size):
            if on_progress:
                on_progress(dest.stat().st_size, dest.stat().st_size, 0, 0)
            return
        source = cached_file(repo, path, revision, expected_size)
        if source is not None:
            reuse_cache(source, dest, on_progress, should_cancel)
            return
        _download_file(cfg, repo, path, dest, on_progress, should_cancel,
                       revision)
        if not (should_cancel and should_cancel()) and \
                not finished_file(dest, expected_size):
            raise RuntimeError(f"{dest.name} does not match the expected file "
                               "size or is an unfinished Git LFS download.")
    finally:
        with _writing_lock:
            _writing.discard(dest)


def _download_file(cfg: dict, repo: str, path: str, dest: Path,
                   on_progress=None, should_cancel=None,
                   revision: str = "main") -> None:
    """Resumable: a .part file, Range on retry, an atomic swap at the end."""
    import urllib.parse
    url = (f"{hf_endpoint(cfg)}/{repo}/resolve/{_rev(revision)}/"
           + urllib.parse.quote(path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(hf_headers(cfg))
    if have:
        headers["Range"] = f"bytes={have}-"
    total = 0
    with requests.get(url, headers=headers, stream=True, timeout=60,
                      allow_redirects=True) as r:
        if r.status_code == 416:
            remote = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
            if remote.isdigit() and int(remote) == have:
                part.replace(dest)
                return
            part.unlink(missing_ok=True)
            r.close()
            return _download_file(cfg, repo, path, dest, on_progress,
                                  should_cancel, revision)
        if r.status_code in (401, 403):
            raise RuntimeError(f"HuggingFace refused {path}. Accept the "
                               "repo's terms on its page and add a token on "
                               "the Models page.")
        if r.status_code == 404:
            raise RuntimeError(f"{repo} has no file {path} (revision "
                               f"{revision}).")
        r.raise_for_status()
        mode = "ab" if (have and r.status_code == 206) else "wb"
        if mode == "wb":
            have = 0
        length = int(r.headers.get("Content-Length", 0))
        total = length + have if length else 0
        got, last, started = have, 0.0, time.time()
        with open(part, mode) as fh:
            # 64 kB: a dropped connection loses at most this much of what
            # already arrived, and the next try resumes after the rest
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if should_cancel and should_cancel():
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if on_progress and now - last > 0.6:
                    last = now
                    speed = (got - have) / max(now - started, .1)
                    eta = (total - got) / speed if speed > 0 and total else 0
                    on_progress(got, total, speed, eta)
    if total and part.stat().st_size != total:
        raise RuntimeError(f"{dest.name} stopped short at "
                           f"{fmt_size(part.stat().st_size)} of "
                           f"{fmt_size(total)} — try again to resume.")
    part.replace(dest)


def expand(cfg: dict, items: list[dict], log=None) -> list[dict]:
    """Turn the set into concrete files: a folder item becomes every file
    under it, and every size becomes the repo's own number where it can be
    read. A repo that cannot be listed keeps the estimates."""
    trees: dict[tuple[str, str], list[dict] | None] = {}
    out: list[dict] = []
    for item in items:
        cached = cached_items(item)
        if cached is not None:
            out.extend(cached)
            if log:
                log(f"Using Hugging Face cache: {item['repo']}/{item['path']}")
            continue
        key = (item["repo"], item["revision"])
        if key not in trees:
            try:
                trees[key] = hf_tree(cfg, *key)
            except Exception as exc:  # noqa: BLE001
                trees[key] = None
                if log:
                    log(f"Could not list {item['repo']} ({exc}); using the "
                        "estimated sizes.")
        tree = trees[key]
        if item.get("prefix"):
            if tree is None:
                raise RuntimeError(f"{item['repo']} could not be listed, so "
                                   f"the files in {item['path']} are unknown.")
            for f in tree:
                if f["path"].startswith(item["path"]):
                    out.append({**item, "path": f["path"], "size": f["size"],
                                "prefix": False, "approx": False,
                                "known_size": f["size"],
                                "name": f["path"].rsplit("/", 1)[-1]})
            continue
        size = next((f["size"] for f in tree or [] if f["path"] == item["path"]),
                    None)
        out.append({**item, "size": size if size is not None else item["size"],
                    "approx": size is None and item.get("approx", False),
                    "known_size": size or 0})
    return out


# --------------------------------------------------------------------------- #
# setup run
# --------------------------------------------------------------------------- #
ENGINE_CHECK = (
    "import importlib.util,json,sys;"
    "mods=['torch','torchvision','transformers','diffusers','optimum.quanto',"
    "'librosa','soundfile','pyloudnorm','einops','easydict','imageio',"
    "'imageio_ffmpeg','skimage','cv2','misaki','ftfy','safetensors'];"
    "bad=[];"
    "[bad.append(m) for m in mods if importlib.util.find_spec(m.split('.')[0]) is None"
    " or (m=='optimum.quanto' and importlib.util.find_spec(m) is None)];"
    "import torch;"
    "print(json.dumps({'missing':bad,'torch':torch.__version__,"
    "'cuda':torch.cuda.is_available(),"
    "'build':(torch.version.cuda or ''),"
    "'xformers':importlib.util.find_spec('xformers') is not None}))"
)


def check_engine(python: str) -> dict:
    """What the engine environment has, by asking it."""
    if not python or not Path(python).exists():
        return {"ok": False, "error": "No engine environment yet."}
    try:
        out = _run([python, "-c", ENGINE_CHECK], timeout=180)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    if out.returncode != 0:
        tail = (out.stderr or out.stdout).strip().splitlines()[-1:] or [""]
        return {"ok": False, "error": tail[0][:300]}
    try:
        d = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": out.stdout[-300:]}
    d["ok"] = not d["missing"]
    return d


def check_engine_loads(python: str, eng: Path) -> tuple[bool, str]:
    """Import the real engine in its environment, as a render would, and run
    its argument parser. Finding the packages is not enough: an engine file
    that will not import (a name a newer Python removed, say) only shows
    here. Needs no GPU."""
    if not (eng / "generate_multitalk.py").exists():
        return False, f"No generate_multitalk.py in {eng}."
    try:
        out = _run([python, "generate_multitalk.py", "--help"], cwd=str(eng),
                   timeout=600, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    if out.returncode == 0 and "--vae_tile" in out.stdout:
        return True, ""
    lines = [ln for ln in (out.stderr or out.stdout).strip().splitlines()
             if ln.strip() and not ln.startswith(" ")]
    return False, (lines[-1] if lines else f"exit code {out.returncode}")[:300]


def make_venv(python: str, log) -> Path:
    vpy = venv_python()
    if vpy.exists():
        ok = _run([str(vpy), "-m", "pip", "--version"], timeout=60)
        if ok.returncode == 0:
            return vpy
        log("The engine environment is broken (no pip) — rebuilding it.")
        shutil.rmtree(venv_dir(), ignore_errors=True)
    res = _run([python, "-m", "venv", str(venv_dir())], timeout=600)
    if res.returncode != 0 or not vpy.exists():
        raise RuntimeError("Could not create the engine environment: "
                           + (res.stderr or res.stdout)[-400:]
                           + (" On Debian/Ubuntu install python3-venv."
                              if platform.system() == "Linux" else ""))
    return vpy


def torch_build(python: str) -> tuple[str, str]:
    """(version, CUDA build) of the torch in that environment, read from
    torch/version.py by its own interpreter without importing torch — or
    ("", "") when there is none. The build is "" for a CPU wheel."""
    code = ("import importlib.util,pathlib,re;"
            "s=importlib.util.find_spec('torch');"
            "t=pathlib.Path(s.origin).with_name('version.py').read_text() if s else '';"
            "v=re.search(r\"__version__\\s*=\\s*['\\\"]([^'\\\"]+)\",t);"
            "c=re.search(r\"cuda\\s*(?::\\s*[^=]+)?=\\s*['\\\"]([^'\\\"]*)\",t);"
            "print((v.group(1) if v else '')+'|'+(c.group(1) if c else ''))")
    try:
        out = _run([python, "-c", code], timeout=60)
        if out.returncode == 0 and "|" in out.stdout:
            version, build = out.stdout.strip().splitlines()[-1].split("|", 1)
            return version, build
    except Exception:  # noqa: BLE001
        pass
    return "", ""


def gpu_processes() -> list[dict]:
    """Programs holding memory on the card, from nvidia-smi. Out of memory on
    8 GB is very often someone else's — a browser, a game launcher — and
    "close something" helps nobody who cannot see what."""
    exe = shutil.which("nvidia-smi") or (
        r"C:\Windows\System32\nvidia-smi.exe"
        if platform.system() == "Windows" else "")
    if not exe:
        return []
    try:
        out = _run([exe, "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"], timeout=20)
    except Exception:  # noqa: BLE001
        return []
    found = []
    for line in out.stdout.splitlines():
        # split on the first and last comma: a path can hold one
        head, _, mem = line.rpartition(",")
        pid, _, name = head.partition(",")
        try:
            # either slash: nvidia-smi names Windows paths with backslashes
            short = name.strip().replace("\\", "/").rsplit("/", 1)[-1]
            found.append({"pid": int(pid), "name": short,
                          "mb": int(float(mem))})
        except ValueError:
            continue
    return found


def wants_cuda(index: str) -> bool:
    return "/cu" in (index or "")


def install_torch(cfg: dict, python: str, log, on_pct=None,
                  should_cancel=None) -> None:
    index = torch_index(cfg)
    # pip counts torch 2.4.1+cpu as satisfying torch==2.4.1, so asking for
    # the CUDA build over a CPU one changes nothing — the trap the sibling
    # studios hit with an RTX 4060 that kept the CPU wheel through every
    # Reinstall. A build that does not match the index is removed first.
    version, build = torch_build(python)
    if version and wants_cuda(index) and not build:
        log(f"torch {version} is the CPU build; removing it so the CUDA "
            "build can take its place.")
        if stream([python, "-m", "pip", "uninstall", "-y", "torch",
                   "torchvision", "torchaudio"], lambda ln: log(ln[:200]),
                  should_cancel=should_cancel) != 0:
            raise RuntimeError("Could not remove the CPU build of PyTorch — "
                               "close anything using the engine environment "
                               "and press Reinstall again.")
    args = list(TORCH_SPEC) + (["--index-url", index] if index else [])
    pip_install(python, args, log, on_pct, should_cancel)
    version, build = torch_build(python)
    if wants_cuda(index) and version and not build:
        raise RuntimeError(f"pip left torch {version}, a CPU build, in place "
                           "of the CUDA build — press Reinstall again.")
    if cfg.get("want_xformers", True):
        # optional: the engine falls back to PyTorch attention without it.
        # --no-deps, or a mismatched wheel would drag torch to another build
        try:
            pip_install(python, [XFORMERS_SPEC, "--no-deps"]
                        + (["--index-url", index] if index else []),
                        log, on_pct, should_cancel)
        except Exception as exc:  # noqa: BLE001
            log(f"xformers was skipped ({exc}); the engine uses PyTorch "
                "attention instead, which works the same, a little slower.")
    forget_gpu()


def install_packages(python: str, log, on_pct=None,
                     should_cancel=None) -> None:
    pip_install(python, ["-r", str(ENGINE_REQUIREMENTS)], log, on_pct,
                should_cancel)


def run_setup(cfg: dict, prog: Progress) -> None:
    prog.running, prog.done, prog.error = True, False, None
    try:
        prog.begin("python")
        eng = engine_dir(cfg)
        if not (eng / "generate_multitalk.py").exists():
            raise RuntimeError(f"The MultiTalk engine is not at {eng}. It ships "
                               "in this repository's MultiTalk folder.")
        base = find_python(prog)
        prog.finish("python", base)

        prog.begin("venv", "Creating the environment…")
        vpy = make_venv(base, prog.log)
        cfg["python"] = str(vpy)
        save_config(cfg)
        prog.track("venv", None, "Upgrading pip…")
        pip_install(str(vpy), ["--upgrade", "pip", "wheel", "setuptools"],
                    prog.log)
        prog.finish("venv", str(vpy))

        prog.begin("torch")
        have = check_engine(str(vpy))
        # the build, not cuda.is_available(): a card the driver hides still
        # wants the CUDA wheel, and a CPU wheel must not be kept (rule above)
        if have.get("torch", "").startswith("2.4.1") and (
                have.get("build") or not wants_cuda(torch_index(cfg))):
            prog.finish("torch", f"torch {have['torch']} already installed")
        else:
            prog.track("torch", None, "Installing PyTorch — the long one…")
            install_torch(cfg, str(vpy), prog.log,
                          lambda pct, d: prog.track("torch", pct, d))
            prog.finish("torch", "torch " + TORCH_SPEC[0].split("==")[1])

        prog.begin("packages", "Installing the engine's packages…")
        install_packages(str(vpy), prog.log,
                         lambda pct, d: prog.track("packages", pct, d))
        prog.finish("packages", "Installed")

        prog.begin("models")
        wdir = weights_dir(cfg)
        todo = [m for m in model_set(cfg) if not present(wdir, m)]
        if not todo:
            prog.finish("models", "Everything is already downloaded")
        else:
            for note in preflight(cfg)["notes"]:
                prog.log("Preflight: " + note)
            prog.track("models", None, "Asking HuggingFace for the file list…")
            plan = [f for f in expand(cfg, todo, prog.log)
                    if not finished_file(model_path(wdir, f), f.get("known_size", 0))]
            grand = sum(f["size"] for f in plan)
            prog.log(f"{len(plan)} file(s) to download, ~{fmt_size(grand)}")
            done_bytes = 0
            for i, f in enumerate(plan, 1):
                dest = model_path(wdir, f)
                head = f"{f['name']} ({i} of {len(plan)})"

                def on_prog(got, total, speed, eta, _head=head, _base=done_bytes):
                    whole = ((_base + got) / grand * 100) if grand else None
                    prog.track("models", whole,
                               f"{_head} — {fmt_transfer(got, total, speed, eta)}")

                prog.track("models", (done_bytes / grand * 100) if grand else None,
                           f"{head} — starting…")
                try:
                    download_file(cfg, f["repo"], f["path"], dest, on_prog,
                                  revision=f["revision"],
                                  expected_size=f.get("known_size", 0))
                except Exception as exc:  # noqa: BLE001
                    if f["role"] != "optional":
                        raise
                    prog.log(f"Skipped {f['name']} ({exc}) — clips still "
                             "render from audio files, without typed speech.")
                done_bytes += f["size"] or (dest.stat().st_size
                                            if dest.exists() else 0)
                prog.log(f"Downloaded {f['name']}")
            prog.finish("models", f"{len(plan)} file(s) ready")

        prog.begin("check", "Asking the engine environment what it has…")
        report = check_engine(str(vpy))
        if not report.get("ok"):
            raise RuntimeError("The engine environment is missing: "
                               + ", ".join(report.get("missing") or [])
                               + (report.get("error") or ""))
        prog.track("check", None, "Loading the engine's code…")
        loads, why = check_engine_loads(str(vpy), eng)
        if not loads:
            raise RuntimeError("The engine's code would not load: " + why)
        prog.finish("check", f"torch {report['torch']}"
                    + (" · CUDA" if report.get("cuda") else " · no GPU found")
                    + (" · xformers" if report.get("xformers") else ""))
        cfg["setup_complete"] = True
        save_config(cfg)
        prog.done = True
        prog.log("Setup complete.")
    except Exception as exc:  # noqa: BLE001
        prog.error = str(exc)
        if prog.step:
            prog.fail(prog.step, str(exc))
        prog.log(f"FAILED: {exc}")
    finally:
        prog.running = False
