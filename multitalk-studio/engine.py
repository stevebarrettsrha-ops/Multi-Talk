"""
engine.py - from a request on the page to a MultiTalk run, and back.

Everything here is a pure function: validate the settings, write the input
JSON generate_multitalk.py reads, build its command line, and turn the lines
it prints into a stage and a percentage. The server owns the process; this
module owns the meaning, which is why it is the part with unit tests.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import bootstrap

FPS = 25                 # MultiTalk is trained at 25 fps
FRAME_NUM = 81           # one clip: 81 frames, 3.24 s
MOTION_FRAME = 25        # frames carried into the next clip when streaming
MAX_STREAM_FRAMES = 1000  # the engine's own cap in streaming mode (40 s)

# What a finished clip measures. MultiTalk renders at a size bucket that
# fits the card (multiples of 32; 360 is not one, and 720 x 1280 natively
# wants ~33 GB of VRAM), so each output picks the bucket of its own shape
# and the studio resizes the result to the exact pixels afterwards.
OUTPUTS = {
    "720x360": {"label": "720 × 360 — landscape", "w": 720, "h": 360,
                "ratio": 360 / 720},
    "720x1280": {"label": "720 × 1280 — portrait", "w": 720, "h": 1280,
                 "ratio": 1280 / 720},
}
DEFAULT_OUTPUT = "720x360"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac",
             ".mp4", ".mov", ".mkv", ".avi"}


class BadRequest(ValueError):
    """A setting the page sent that cannot be used, with advice."""


def clips_for(frames: int, mode: str = "streaming") -> int:
    """How many clips the engine renders for `frames` of audio — the same
    loop arithmetic as MultiTalkPipeline.generate()."""
    if mode == "clip":
        return 1
    target = min(MAX_STREAM_FRAMES, max(int(frames), FRAME_NUM + 1))
    step = FRAME_NUM - MOTION_FRAME
    return 1 + max(1, math.ceil((target - FRAME_NUM) / step))


def video_seconds(audio_seconds: float, mode: str = "streaming") -> float:
    """What comes out: one 3.24 s clip, or the audio's length up to 40 s."""
    if mode == "clip":
        return FRAME_NUM / FPS
    return min(max(audio_seconds, (FRAME_NUM + 4) / FPS), MAX_STREAM_FRAMES / FPS)


def _num(params: dict, key: str, default, lo, hi, cast=float):
    raw = params.get(key, default)
    if raw is None or raw == "":
        raw = default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"{key.replace('_', ' ').capitalize()} must be a "
                         "number.") from None
    if isinstance(value, float) and not math.isfinite(value):
        raise BadRequest(f"{key} must be a finite number.")
    if not lo <= value <= hi:
        raise BadRequest(f"{key.replace('_', ' ').capitalize()} must be "
                         f"between {lo} and {hi}.")
    return value


TTS_TAG = re.compile(r"\(s(\d+)\)")


def normalize(params: dict, cfg: dict) -> dict:
    """Check and complete a request. Raises BadRequest with advice."""
    if not isinstance(params, dict):
        raise BadRequest("Send the settings as an object.")
    p = bootstrap.PRECISIONS.get(cfg.get("precision") or "",
                                 bootstrap.PRECISIONS["int8-fusionx"])
    d = p["defaults"]
    out: dict = {}

    image = params.get("image") or ""
    if not isinstance(image, str) or not image:
        raise BadRequest("Add a reference picture — MultiTalk animates the "
                         "people in it.")
    out["image"] = image

    people = params.get("people", 1)
    try:
        people = int(people)
    except (TypeError, ValueError):
        raise BadRequest("People must be 1 or 2.") from None
    if people not in (1, 2):
        raise BadRequest("MultiTalk drives one or two people.")
    out["people"] = people

    source = params.get("source") or "files"
    if source not in ("files", "tts"):
        raise BadRequest("Audio comes from files or from typed speech (tts).")
    out["source"] = source
    if source == "files":
        a1, a2 = params.get("audio1") or "", params.get("audio2") or ""
        if not isinstance(a1, str) or not isinstance(a2, str):
            raise BadRequest("Audio names must be text.")
        if people == 1 and not a1:
            raise BadRequest("Add the voice recording for the person.")
        if people == 2 and not (a1 or a2):
            raise BadRequest("Add a recording for at least one of the two "
                             "people.")
        out["audio1"], out["audio2"] = a1, (a2 if people == 2 else "")
        order = params.get("audio_type") or "add"
        if order not in ("add", "para"):
            raise BadRequest("Two voices either take turns (add) or speak at "
                             "once (para).")
        out["audio_type"] = order
    else:
        text = str(params.get("tts_text") or "").strip()
        if not text:
            raise BadRequest("Type what the person says.")
        if len(text) > 4000:
            raise BadRequest("Keep the speech under 4000 characters.")
        v1, v2 = params.get("voice1") or "", params.get("voice2") or ""
        if not v1 or (people == 2 and not v2):
            raise BadRequest("Pick a voice for " + ("each person." if people == 2
                                                    else "the person."))
        if people == 2:
            tags = set(TTS_TAG.findall(text))
            if not tags:
                raise BadRequest("For two people, mark who speaks: "
                                 "(s1) Hello there. (s2) Hi!")
            if not tags <= {"1", "2"}:
                raise BadRequest("Only (s1) and (s2) are speakers here.")
        out.update(tts_text=text, voice1=str(v1),
                   voice2=str(v2) if people == 2 else "", audio_type="add")

    prompt = str(params.get("prompt") or "").strip()
    if len(prompt) > 2000:
        raise BadRequest("Keep the prompt under 2000 characters.")
    out["prompt"] = prompt or ("A person is talking." if people == 1 else
                               "Two people are having a conversation.")

    size = params.get("size") or "multitalk-360"
    if size not in bootstrap.SIZES:
        raise BadRequest("Size must be one of " + ", ".join(
            v["label"] for v in bootstrap.SIZES.values()) + ".")
    out["size"] = size
    output = params.get("output") or DEFAULT_OUTPUT
    if output not in OUTPUTS:
        raise BadRequest("The video is 720 × 360 (landscape) or 720 × 1280 "
                         "(portrait).")
    out["output"] = output
    mode = params.get("mode") or "streaming"
    if mode not in ("clip", "streaming"):
        raise BadRequest("Length is either one clip or the whole audio.")
    out["mode"] = mode

    out["steps"] = _num(params, "steps", d["steps"], 2, 60, int)
    out["text_scale"] = _num(params, "text_scale", d["text_scale"], 0, 15)
    out["audio_scale"] = _num(params, "audio_scale", d["audio_scale"], 0, 15)
    out["shift"] = _num(params, "shift", d["shift"], 0.5, 20)
    seed = params.get("seed")
    out["seed"] = -1 if seed in (None, "", -1) else _num(params, "seed", -1, 0,
                                                         2 ** 31 - 1, int)
    out["teacache"] = bool(params.get("teacache", d["teacache"]))
    out["teacache_thresh"] = _num(params, "teacache_thresh", 0.3, 0.05, 1.0)
    out["apg"] = bool(params.get("apg", False))
    out["color_correction"] = _num(params, "color_correction", 1.0, 0, 1)
    out["t5_cpu"] = bool(params.get("t5_cpu", True))
    vae_tile = _num(params, "vae_tile", 32, 0, 128, int)
    if vae_tile and vae_tile < 16:
        raise BadRequest("VAE tiles under 16 latent pixels only add seams; "
                         "use 0 to turn tiling off.")
    out["vae_tile"] = vae_tile
    out["persistent"] = _num(params, "persistent", 0, 0, 20_000_000_000, int)
    out["title"] = str(params.get("title") or "")[:120]
    return out


def resize_args(ffmpeg: str, src: Path, dst: Path, w: int, h: int,
                scale: float) -> list[str]:
    """ffmpeg's command to bring a render to exactly w x h: scaled to cover
    (lanczos), centre-cropped, a light sharpen where it was enlarged a lot,
    the soundtrack copied as it is."""
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={w}:{h},setsar=1")
    if scale > 1.4:
        vf += ",unsharp=5:5:0.6:5:5:0.0"
    return [ffmpeg, "-v", "error", "-y", "-i", str(src), "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
            str(dst)]


def as_path(p: Path) -> str:
    """Forward slashes everywhere: generate_multitalk.py names its audio
    folder after cond_image.split('/')[-1], which a Windows backslash path
    would turn into the whole path."""
    return Path(p).resolve().as_posix()


def input_json(job: dict, uploads: Path, voices: Path, audio_dir: Path) -> dict:
    """The meta file generate_multitalk.py --input_json reads."""
    data = {"prompt": job["prompt"],
            "cond_image": as_path(uploads / job["image"])}
    if job["source"] == "files":
        if job["people"] == 1:
            data["cond_audio"] = {"person1": as_path(uploads / job["audio1"])}
        else:
            data["audio_type"] = job["audio_type"]
            data["cond_audio"] = {
                "person1": as_path(uploads / job["audio1"]) if job["audio1"]
                else "None",
                "person2": as_path(uploads / job["audio2"]) if job["audio2"]
                else "None"}
    else:
        tts = {"text": job["tts_text"],
               "human1_voice": as_path(voices / f"{job['voice1']}.pt")}
        if job["people"] == 2:
            tts["human2_voice"] = as_path(voices / f"{job['voice2']}.pt")
            data["audio_type"] = "add"
        data["tts_audio"] = tts
        data["cond_audio"] = {}
    return data


def argv(cfg: dict, job: dict, json_path: Path, audio_dir: Path,
         save_file: Path) -> list[str]:
    """generate_multitalk.py's command line for this job."""
    wdir = bootstrap.weights_dir(cfg)
    precision = cfg.get("precision") or "int8-fusionx"
    p = bootstrap.PRECISIONS.get(precision, bootstrap.PRECISIONS["int8-fusionx"])
    cmd = [bootstrap.engine_python(cfg) or "python", "generate_multitalk.py",
           "--ckpt_dir", as_path(wdir / bootstrap.WAN_DIR),
           "--wav2vec_dir", as_path(wdir / bootstrap.WAV2VEC_DIR),
           "--kokoro_dir", as_path(wdir / bootstrap.KOKORO_DIR),
           "--quant", "int8",
           "--quant_dir", as_path(wdir / bootstrap.MULTITALK_DIR),
           "--input_json", as_path(json_path),
           "--audio_save_dir", as_path(audio_dir),
           "--save_file", as_path(save_file),
           "--size", job["size"],
           "--bucket_ratio", f"{OUTPUTS[job.get('output') or DEFAULT_OUTPUT]['ratio']:.4f}",
           "--mode", job["mode"],
           "--frame_num", str(FRAME_NUM),
           "--motion_frame", str(MOTION_FRAME),
           "--sample_steps", str(job["steps"]),
           "--sample_shift", f"{job['shift']:g}",
           "--sample_text_guide_scale", f"{job['text_scale']:g}",
           "--sample_audio_guide_scale", f"{job['audio_scale']:g}",
           "--color_correction_strength", f"{job['color_correction']:g}",
           "--num_persistent_param_in_dit", str(job["persistent"]),
           "--offload_model", "True",
           "--vae_tile", str(job["vae_tile"]),
           "--vae_tile_overlap", "8",
           "--audio_mode", "tts" if job["source"] == "tts" else "localfile"]
    if precision == "int8-fusionx":
        # the FusionX build is a whole INT8 model, handed over as the "LoRA";
        # its quantisation map is read from beside it
        cmd += ["--lora_dir", as_path(wdir / bootstrap.MULTITALK_DIR
                                      / p["dit"]["path"])]
    if job["seed"] >= 0:
        cmd += ["--base_seed", str(job["seed"])]
    else:
        cmd += ["--base_seed", "-1"]
    if job["t5_cpu"]:
        cmd += ["--t5_cpu"]
    if job["teacache"]:
        cmd += ["--use_teacache", "--teacache_thresh", f"{job['teacache_thresh']:g}"]
    if job["apg"]:
        cmd += ["--use_apg"]
    return cmd


# --------------------------------------------------------------------------- #
# reading the engine's output
# --------------------------------------------------------------------------- #
CLIPS = re.compile(r"\[clips\] total=(\d+) frames=(\d+)")
CLIP = re.compile(r"\[clip\] (\d+)")
TQDM = re.compile(r"(\d+)/(\d+) \[")
SEED = re.compile(r"base_seed=(-?\d+)")


def new_state() -> dict:
    return {"stage": "Starting the engine", "pct": 1.0, "clips": 0, "clip": 0,
            "step": 0, "steps": 0, "frames": 0, "oom": False, "saving": False,
            "tail": [], "seed": None, "phase": ""}


def read_line(state: dict, line: str) -> None:
    """Fold one line of engine output into the job's state."""
    tail = state["tail"]
    tail.append(line[:300])
    if len(tail) > 60:
        del tail[:20]
    low = line.lower()
    if "out of memory" in low or "outofmemoryerror" in low:
        state["oom"] = True
    m = SEED.search(line)
    if m and state["seed"] is None and "Generation job args" in line:
        state["seed"] = int(m.group(1))
    if "Creating MultiTalk pipeline" in line:
        state.update(stage="Loading the model into RAM", pct=max(state["pct"], 5))
    elif "Loading Quantized" in line or "loading " in line:
        state.update(stage="Loading the model into RAM", pct=max(state["pct"], 6))
    elif "Generating video" in line:
        state.update(stage="Reading the prompt and picture",
                     pct=max(state["pct"], 14))
    m = CLIPS.search(line)
    if m:
        state["clips"], state["frames"] = int(m.group(1)), int(m.group(2))
    m = CLIP.search(line)
    if m:
        state["clip"], state["step"], state["phase"] = int(m.group(1)), 0, ""
        state["stage"] = _clip_stage(state)
    elif "[clip] encoding" in line or "[clip] decoding" in line:
        # the VAE stretches either side of sampling: minutes on a CPU,
        # tens of seconds on a 4060 — named, so the bar is not "stuck"
        state["phase"] = ("reading the picture" if "encoding" in line
                          else "decoding the frames")
        state["stage"] = _clip_stage(state)
    if "Saving video" in line or "Saving generated video" in line:
        state["saving"] = True
        state.update(stage="Writing the video", pct=max(state["pct"], 96))
        return
    m = TQDM.search(line)
    if m and not state["saving"] and state["clip"]:
        step, steps = int(m.group(1)), int(m.group(2))
        state["step"], state["steps"] = step, steps
        state["phase"] = ""
        state["stage"] = _clip_stage(state)
        state["pct"] = max(state["pct"], sampling_pct(state))
    if line.strip().endswith("Finished."):
        state.update(stage="Finished", pct=99)


def _clip_stage(state: dict) -> str:
    clips = state["clips"] or 1
    head = (f"Clip {state['clip']} of {clips}" if clips > 1
            else "Rendering")
    if state.get("phase"):
        return f"{head} · {state['phase']}"
    if state["steps"] and state["step"]:
        return f"{head} · step {state['step']} of {state['steps']}"
    return head


def sampling_pct(state: dict) -> float:
    """15-95 % across every step of every clip."""
    clips = max(state["clips"], state["clip"], 1)
    steps = max(state["steps"], 1)
    done = (state["clip"] - 1) + min(state["step"] / steps, 1)
    return round(15 + 80 * min(done / clips, 1), 1)


def failure(state: dict, code: int) -> str:
    """The sentence a person sees when the engine stopped."""
    if state["oom"]:
        return ("Out of GPU memory. Pick a smaller size, keep VAE tiling on "
                "and the text encoder on the CPU, and close anything else "
                "using the GPU.")
    text = "\n".join(state["tail"])
    if "Torch not compiled with CUDA enabled" in text:
        return ("PyTorch in the engine environment is the CPU build. On the "
                "Engine page press Reinstall next to PyTorch to get the CUDA "
                "build.")
    if ("Found no NVIDIA driver" in text or "no CUDA GPUs are available" in text
            or "CUDA driver version is insufficient" in text):
        return ("PyTorch cannot reach the graphics card: the NVIDIA driver is "
                "missing or too old for CUDA 12.1. Install the current driver "
                "from nvidia.com, restart, then try again.")
    if "No such file or directory" in text or "FileNotFoundError" in text:
        hit = next((ln for ln in reversed(state["tail"])
                    if "No such file" in ln or "FileNotFoundError" in ln), "")
        return "A file the engine needs is missing: " + hit[-220:]
    if "ModuleNotFoundError" in text:
        hit = next((ln for ln in reversed(state["tail"])
                    if "ModuleNotFoundError" in ln), "")
        return (hit[-160:] + " — run Install on the Engine page to finish the "
                "engine environment.")
    if "Aduio file not exists" in text:
        return "The engine could not read the audio. Try a WAV file."
    last = next((ln for ln in reversed(state["tail"])
                 if "Error" in ln or "error" in ln), "")
    return (last[-300:] or f"The engine stopped (exit code {code}). The log "
            "on the job has the details.")
