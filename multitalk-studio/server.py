"""
server.py - MultiTalk Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7806)

Each render is one run of MultiTalk's own generate_multitalk.py in the engine
environment, one at a time — an 8 GB card has room for exactly one — with
its output read line by line for the progress bar. Nothing of the model ever
loads into this process, so a crash or an out-of-memory stop in the engine
never takes the page down with it.
"""

from __future__ import annotations

import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

import bootstrap
import engine
import manager
import selftest
from bootstrap import APP_DIR, Progress, load_config, save_config

DATA_DIR = bootstrap.DATA_DIR
CLIPS_DIR = DATA_DIR / "clips"
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
GALLERY_PATH = DATA_DIR / "gallery.json"
WEB_DIR = APP_DIR / "web"
PORT = int(os.environ.get("MULTITALK_STUDIO_PORT", "7806"))

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

cfg = load_config()
progress = Progress()
setup_lock = threading.Lock()

# set while the start-up search walks the drives, so the Engine page says
# "searching" instead of "missing" and Recheck does not start a second walk
locating = threading.Event()
_locate_lock = threading.Lock()


def _say(msg: str) -> None:
    print(f"[multitalk-studio] {msg}", flush=True)


def _heal(search: bool = False, force: bool = False) -> None:
    """Verify the saved locations; repair any that moved.

    Without `search` only the quick repair runs (a moved repo folder). With
    it, an engine that is still nowhere is searched for across the drives.
    MULTITALK_STUDIO_NO_SEARCH=1 turns all of it off: the tests' configs
    point at made-up folders on purpose, and must not adopt a real checkout.
    """
    if os.environ.get("MULTITALK_STUDIO_NO_SEARCH") == "1":
        locating.clear()
        return
    # a quick repair skips a turn when another check holds the lock; a search
    # waits for it (it is owed: `locating` is already set for it)
    if not _locate_lock.acquire(blocking=search):
        return
    try:
        if search:
            locating.set()
        before = dict(cfg)
        notes = bootstrap.verify_locations(cfg, search=search, log=_say,
                                           force=force)
        if cfg != before:
            save_config(cfg)
        if notes:
            manager.forget()
            for n in notes:
                _say(n)
        if search:
            for line in bootstrap.location_report(cfg):
                _say("Verified " + line)
    finally:
        if search:
            locating.clear()
        _locate_lock.release()


_heal()

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
gallery_lock = threading.Lock()
wake = threading.Event()
JOB_PUBLIC = ("id", "status", "pct", "stage", "created", "started", "finished",
              "title", "error", "clip", "position", "settings")


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def _read_gallery_unlocked() -> list[dict]:
    if not GALLERY_PATH.exists():
        return []
    try:
        value = json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("not a list")
    except (OSError, ValueError):
        bootstrap.quarantine(GALLERY_PATH)
        return []
    return [v for v in value if isinstance(v, dict) and v.get("id")
            and v.get("file")]


def read_gallery() -> list[dict]:
    with gallery_lock:
        return _read_gallery_unlocked()


def add_clip(item: dict) -> None:
    with gallery_lock:
        bootstrap.atomic_write(GALLERY_PATH, json.dumps(
            [item] + _read_gallery_unlocked(), indent=2))


def title_from(job: dict) -> str:
    text = job.get("title") or job.get("tts_text") or job.get("prompt") or ""
    text = engine.TTS_TAG.sub("", text).strip()
    return " ".join(text.split()[:8]).strip(" ,.!?-") or "Untitled clip"


# --------------------------------------------------------------------------- #
# the render queue
# --------------------------------------------------------------------------- #
def public(job: dict) -> dict:
    return {k: job.get(k) for k in JOB_PUBLIC}


def next_job() -> dict | None:
    with jobs_lock:
        queued = sorted((j for j in jobs.values() if j["status"] == "queued"),
                        key=lambda j: j["created"])
        if not queued:
            return None
        job = queued[0]
        job.update(status="running", started=time.time(),
                   stage="Starting the engine", pct=1)
        return job


def worker() -> None:
    while True:
        job = next_job()
        if not job:
            wake.wait(2)
            wake.clear()
            continue
        try:
            run_job(job)
        except Exception as exc:  # noqa: BLE001
            with jobs_lock:
                job.update(status="error", stage="Failed", finished=time.time(),
                           error=f"{type(exc).__name__}: {exc}")


def run_job(job: dict) -> None:
    settings = job["settings"]
    jdir = JOBS_DIR / job["id"]
    audio_dir = jdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    voices = bootstrap.weights_dir(cfg) / bootstrap.KOKORO_DIR / "voices"
    meta = engine.input_json(settings, UPLOADS_DIR, voices, audio_dir)
    json_path = jdir / "input.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    save_file = CLIPS_DIR / job["id"]
    cmd = engine.argv(cfg, settings, json_path, audio_dir, save_file)
    state = engine.new_state()
    log_path = jdir / "engine.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           # Windows hands a piped child the ANSI code page; UTF-8 both ends
           "PYTHONUTF8": "1",
           # the multi-GPU path is never taken; keep tokenizers quiet
           "TOKENIZERS_PARALLELISM": "false"}
    if platform.system() != "Windows":
        # fragments from the per-layer offload otherwise pile up on 8 GB
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # line-buffered: a render the app never sees finish (closed, crashed)
    # still leaves its log behind for the person reading why
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write("$ " + " ".join(cmd) + "\n")

        def on_line(line: str) -> None:
            log.write(line + "\n")
            engine.read_line(state, line)
            with jobs_lock:
                job["stage"], job["pct"] = state["stage"], state["pct"]
                job["log"].append(line[:300])
                if len(job["log"]) > 400:
                    del job["log"][:100]

        def on_start(proc) -> None:
            with jobs_lock:
                job["proc"] = proc

        code = bootstrap.stream(cmd, on_line,
                                cwd=str(bootstrap.engine_dir(cfg)), env=env,
                                should_cancel=lambda: job.get("cancelled"),
                                on_start=on_start)
    with jobs_lock:
        job["proc"] = None
    out = save_file.with_suffix(".mp4")
    if job.get("cancelled"):
        out.unlink(missing_ok=True)
        with jobs_lock:
            job.update(status="cancelled", stage="Stopped", finished=time.time())
        return
    if code != 0 or not out.exists():
        error = engine.failure(state, code)
        if state["oom"]:
            others = [p for p in bootstrap.gpu_processes() if p["mb"] >= 100]
            if others:
                error += " Also on the card right now: " + ", ".join(
                    f"{p['name']} ({p['mb'] / 1024:.1f} GB)" for p in others[:4]) \
                    + "."
        with jobs_lock:
            job.update(status="error", stage="Failed", finished=time.time(),
                       error=error)
        return
    shutil.rmtree(audio_dir, ignore_errors=True)
    # to the exact output size: the render is the bucket that fits the card
    want = engine.OUTPUTS[settings.get("output") or engine.DEFAULT_OUTPUT]
    ffmpeg = selftest.find_ffmpeg(bootstrap.engine_python(cfg))
    render_w, render_h = selftest.video_size(ffmpeg, out) if ffmpeg else (0, 0)
    with jobs_lock:
        job.update(stage=f"Resizing to {want['w']} × {want['h']}",
                   pct=max(job.get("pct") or 0, 97))
    if not ffmpeg or not render_w:
        with jobs_lock:
            job.update(status="error", stage="Failed", finished=time.time(),
                       error="No ffmpeg could read the render to resize it — "
                             "install the engine packages on the Engine page.")
        return
    scale = max(want["w"] / render_w, want["h"] / render_h)
    sized = out.with_name(out.stem + ".sized.mp4")
    done = subprocess.run(engine.resize_args(ffmpeg, out, sized, want["w"],
                                             want["h"], scale),
                          capture_output=True, text=True)
    if done.returncode != 0 or not sized.exists():
        sized.unlink(missing_ok=True)
        with jobs_lock:
            job.update(status="error", stage="Failed", finished=time.time(),
                       error="Resizing the render failed: "
                             + (done.stderr.strip().splitlines() or ["?"])[-1][:200])
        return
    os.replace(sized, out)
    frames = state["frames"] if settings["mode"] == "streaming" else engine.FRAME_NUM
    item = {
        "id": job["id"], "file": out.name, "title": title_from(settings),
        "prompt": settings["prompt"], "image": settings["image"],
        "people": settings["people"], "source": settings["source"],
        "audio1": settings.get("audio1", ""), "audio2": settings.get("audio2", ""),
        "audio_type": settings.get("audio_type", ""),
        "tts_text": settings.get("tts_text", ""),
        "voice1": settings.get("voice1", ""), "voice2": settings.get("voice2", ""),
        "size": settings["size"], "mode": settings["mode"],
        "output": settings.get("output") or engine.DEFAULT_OUTPUT,
        "width": want["w"], "height": want["h"],
        "render_width": render_w, "render_height": render_h,
        "steps": settings["steps"], "text_scale": settings["text_scale"],
        "audio_scale": settings["audio_scale"], "shift": settings["shift"],
        "teacache": settings["teacache"], "vae_tile": settings["vae_tile"],
        "t5_cpu": settings["t5_cpu"],
        "seed": state["seed"] if state["seed"] is not None else settings["seed"],
        "precision": cfg.get("precision"),
        "seconds": round(frames / engine.FPS, 2) if frames else None,
        "took": round(time.time() - job["started"]),
        "created": time.time(),
    }
    add_clip(item)
    with jobs_lock:
        job.update(status="done", stage="Ready", pct=100, finished=time.time(),
                   clip=item)


def engine_blocker() -> str:
    """Why a render cannot start, in a sentence — or "" when it can."""
    if not (bootstrap.engine_dir(cfg) / "generate_multitalk.py").exists():
        return "The MultiTalk engine folder is missing. Check Settings."
    if not bootstrap.engine_python(cfg):
        return "The engine environment is not installed. Run setup."
    missing = bootstrap.missing_models(bootstrap.weights_dir(cfg), cfg)
    if missing:
        return ("Weights missing: " + ", ".join(m["name"] for m in missing)
                + ". Download them on the Models page.")
    return ""


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


@app.before_request
def local_only():
    """Only this machine's own pages may drive the app. Binding to 127.0.0.1
    is not enough: any site the person visits can post to it, and DNS
    rebinding lets one read the answers — and this app runs pip and
    processes."""
    raw = request.host or ""
    host = raw.split("]")[0].lower() + "]" if raw.startswith("[") \
        else raw.rsplit(":", 1)[0].lower()
    if host not in LOCAL_HOSTS:
        return jsonify({"error": "MultiTalk Studio only answers to localhost."}), 403
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/") \
                and origin.rstrip("/") not in (
                    f"http://{h}:{PORT}" for h in LOCAL_HOSTS):
            return jsonify({"error": "Cross-site request refused."}), 403


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


# --------------------------------------------------------------------------- #
# status / setup / config
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status():
    wdir = bootstrap.weights_dir(cfg)
    python = bootstrap.engine_python(cfg)
    vram, gpu = bootstrap._GPU_SEEN.get(python, (0, ""))
    with jobs_lock:
        running = [j["id"] for j in jobs.values() if j["status"] == "running"]
        queued = sum(1 for j in jobs.values() if j["status"] == "queued")
    blocker = engine_blocker()
    return jsonify({
        "setup_complete": bool(cfg.get("setup_complete")),
        "ready": not blocker, "blocker": blocker,
        "missing_models": [m["name"] for m in
                           bootstrap.missing_models(wdir, cfg)],
        "tts_ready": bool(manager.voices(cfg)) and (
            wdir / bootstrap.KOKORO_DIR / "kokoro-v1_0.pth").is_file(),
        "python": python, "gpu": gpu, "vram": vram,
        "recommended": bootstrap.recommended(vram),
        "precisions": {k: {"label": v["label"], "note": v["note"],
                           "defaults": v["defaults"]}
                       for k, v in bootstrap.PRECISIONS.items()},
        "sizes": bootstrap.SIZES,
        "outputs": engine.OUTPUTS,
        "config": {k: cfg.get(k) for k in
                   ("engine_dir", "weights_dir", "precision", "want_tts",
                    "want_xformers", "torch_index")},
        "weights_dir": str(wdir),
        "running": running, "queued": queued,
    })


@app.post("/api/setup/start")
def api_setup_start():
    with setup_lock:
        if progress.running:
            return jsonify({"error": "Setup is already running."}), 409
        progress.__init__()
        progress.running = True
    try:
        b = request.get_json(silent=True) or {}
        _apply_config(b)
        threading.Thread(target=_run_setup, daemon=True).start()
    except Exception:
        progress.running = False
        raise
    return jsonify({"ok": True})


def _run_setup() -> None:
    try:
        bootstrap.run_setup(cfg, progress)
    finally:
        manager.forget()
        threading.Thread(target=_warm_gpu, daemon=True).start()


@app.get("/api/setup/state")
def api_setup_state():
    try:
        since = int(request.args.get("since", 0))
    except ValueError:
        since = 0
    return jsonify(progress.snapshot(since))


def _apply_config(b: dict) -> None:
    if b.get("precision") in bootstrap.PRECISIONS:
        cfg["precision"] = b["precision"]
    for key in ("want_tts", "want_xformers"):
        if key in b:
            cfg[key] = bool(b[key])
    for key in ("engine_dir", "weights_dir", "torch_index"):
        if key in b and isinstance(b[key], str):
            cfg[key] = b[key].strip()
    save_config(cfg)


@app.post("/api/config")
def api_config():
    b = request.get_json(silent=True) or {}
    if not isinstance(b, dict):
        return jsonify({"error": "Send settings as an object."}), 400
    _apply_config(b)
    return jsonify({"ok": True})


@app.get("/api/preflight")
def api_preflight():
    return jsonify(bootstrap.preflight(cfg))


# --------------------------------------------------------------------------- #
# dependencies / tasks
# --------------------------------------------------------------------------- #
@app.get("/api/deps")
def api_deps():
    if request.args.get("fresh"):
        manager.forget()
    if not locating.is_set():
        force = request.args.get("relocate") == "1"
        _heal(force=force)
        if (force or bootstrap.needs_location_search(cfg)) and \
                os.environ.get("MULTITALK_STUDIO_NO_SEARCH") != "1":
            locating.set()
            threading.Thread(target=_heal, args=(True, force), daemon=True).start()
    searching = locating.is_set()
    return jsonify({"items": manager.dependencies(cfg, searching=searching),
                    "searching": searching,
                    "torch_index": cfg.get("torch_index", "")})


@app.post("/api/deps/<dep_id>/install")
def api_dep_install(dep_id: str):
    b = request.get_json(silent=True) or {}
    if isinstance(b.get("torch_index"), str):
        cfg["torch_index"] = b["torch_index"].strip()
        save_config(cfg)
    try:
        return jsonify({"ok": True,
                        "task": manager.install_dependency(dep_id, cfg).view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/tasks")
def api_tasks():
    task_id = request.args.get("id", "")
    try:
        since = int(request.args.get("since", 0))
    except ValueError:
        since = 0
    if task_id:
        task = manager.TASKS.get(task_id)
        if not task:
            return jsonify({"error": "No such task."}), 404
        return jsonify(task.view(since))
    return jsonify([t.view(t.view()["cursor"])
                    for t in manager.TASKS.visible()])


@app.post("/api/tasks/<task_id>/cancel")
def api_task_cancel(task_id: str):
    task = manager.TASKS.get(task_id)
    if not task:
        return jsonify({"error": "No such task."}), 404
    task.cancel = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
@app.get("/api/hf/settings")
def api_hf_settings():
    token = cfg.get("hf_token") or ""
    return jsonify({"endpoint": cfg.get("hf_endpoint") or bootstrap.HF_BASE,
                    "token_set": bool(token),
                    "token_hint": ("…" + token[-4:]) if len(token) > 4 else "",
                    "curated": manager.curated(cfg)})


@app.post("/api/hf/settings")
def api_hf_settings_save():
    b = request.get_json(silent=True) or {}
    if "token" in b:
        cfg["hf_token"] = str(b["token"] or "").strip()
    if isinstance(b.get("endpoint"), str):
        cfg["hf_endpoint"] = b["endpoint"].strip() or bootstrap.HF_BASE
    save_config(cfg)
    return jsonify({"ok": True})


@app.post("/api/hf/download")
def api_hf_download():
    b = request.get_json(silent=True) or {}
    try:
        if b.get("precision") in bootstrap.PRECISIONS:
            cfg["precision"] = b["precision"]
            save_config(cfg)
        tasks = manager.download_set(cfg)
        if not tasks:
            return jsonify({"ok": True, "tasks": [],
                            "note": "Everything in that set is already here."})
        return jsonify({"ok": True, "tasks": [t.view() for t in tasks]})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/hf/local")
def api_hf_local():
    return jsonify({"models": manager.local_models(cfg),
                    "weights_dir": str(bootstrap.weights_dir(cfg))})


@app.delete("/api/hf/local")
def api_hf_delete():
    b = request.get_json(silent=True) or {}
    try:
        manager.delete_model(cfg, str(b.get("path") or ""))
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/voices")
def api_voices():
    return jsonify(manager.voices(cfg))


# --------------------------------------------------------------------------- #
# uploads
# --------------------------------------------------------------------------- #
UPLOAD_NAME = re.compile(r"^[0-9a-f]{12}\.[a-z0-9]{2,5}$")


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file received."}), 400
    ext = Path(f.filename).suffix.lower()
    if ext in engine.IMAGE_EXT:
        kind = "image"
    elif ext in engine.AUDIO_EXT:
        kind = "audio"
    else:
        return jsonify({"error": f"{ext or 'That file'} is neither a picture "
                                 "nor audio this app reads."}), 400
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex[:12] + ext
    f.save(UPLOADS_DIR / name)
    if not (UPLOADS_DIR / name).stat().st_size:
        (UPLOADS_DIR / name).unlink(missing_ok=True)
        return jsonify({"error": "That file is empty."}), 400
    return jsonify({"ok": True, "name": name, "kind": kind,
                    "label": Path(f.filename).name[:60]})


@app.get("/api/upload/<name>")
def api_upload_get(name: str):
    if not UPLOAD_NAME.match(name) or not (UPLOADS_DIR / name).is_file():
        return jsonify({"error": "No such upload."}), 404
    return send_file(UPLOADS_DIR / name, conditional=True)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@app.post("/api/generate")
def api_generate():
    params = request.get_json(silent=True)
    try:
        settings = engine.normalize(params, cfg)
    except engine.BadRequest as exc:
        return jsonify({"error": str(exc)}), 400
    for key in ("image", "audio1", "audio2"):
        name = settings.get(key) or ""
        if name and (not UPLOAD_NAME.match(name)
                     or not (UPLOADS_DIR / name).is_file()):
            return jsonify({"error": f"The {'picture' if key == 'image' else 'recording'}"
                                     " is no longer on disk — add it again."}), 400
    if Path(settings["image"]).suffix not in engine.IMAGE_EXT:
        return jsonify({"error": "The reference must be a picture."}), 400
    if settings["source"] == "tts":
        have = set(manager.voices(cfg))
        for v in (settings["voice1"], settings.get("voice2")):
            if v and v not in have:
                return jsonify({"error": f"No voice called {v}. Download the "
                                         "text-to-speech weights on the Models "
                                         "page."}), 400
    blocker = engine_blocker()
    if blocker:
        return jsonify({"error": blocker}), 503
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "pct": 0,
                        "stage": "Waiting for the GPU", "created": time.time(),
                        "title": title_from(settings), "settings": settings,
                        "log": [], "proc": None}
    wake.set()
    return jsonify({"jobs": [job_id]})


@app.get("/api/jobs")
def api_jobs():
    with jobs_lock:
        cutoff = time.time() - 3600
        for jid in [k for k, j in jobs.items()
                    if j["status"] not in ("running", "queued")
                    and j.get("finished", j["created"]) < cutoff]:
            del jobs[jid]
        active = [j for j in jobs.values()
                  if j["status"] in ("running", "queued")
                  or time.time() - j.get("finished", j["created"]) < 600]
        # how many renders are ahead of each waiting one, the running one
        # included — worked out now, since the queue moves under it
        ahead = sum(1 for j in jobs.values() if j["status"] == "running")
        for j in sorted((j for j in jobs.values() if j["status"] == "queued"),
                        key=lambda j: j["created"]):
            j["position"] = ahead
            ahead += 1
        return jsonify([public(j) for j in
                        sorted(active, key=lambda j: j["created"], reverse=True)])


@app.get("/api/jobs/<job_id>/log")
def api_job_log(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "No such job."}), 404
        return jsonify({"lines": list(job["log"][-200:])})


@app.post("/api/jobs/<job_id>/cancel")
def api_job_cancel(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "No such job."}), 404
        if job["status"] == "queued":
            job.update(status="cancelled", stage="Removed from the queue",
                       finished=time.time())
            return jsonify({"ok": True})
        if job["status"] != "running":
            return jsonify({"error": "That job has already finished."}), 409
        job["cancelled"] = True
        proc = job.get("proc")
    # the reader notices on its next line; a silent engine is stopped here
    if proc is not None:
        threading.Thread(target=bootstrap.kill_tree, args=(proc,),
                         daemon=True).start()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
@app.get("/api/clips")
def api_clips():
    return jsonify(read_gallery())


@app.get("/api/clip/<clip_id>")
def api_clip(clip_id: str):
    for item in read_gallery():
        if item["id"] == clip_id:
            path = CLIPS_DIR / item["file"]
            if not path.exists():
                return jsonify({"error": "That file is missing."}), 404
            name = " ".join(str(item.get("title") or "clip").split())[:120]
            return send_file(path, mimetype=mimetypes.guess_type(path.name)[0]
                             or "video/mp4", conditional=True,
                             download_name=f"{name or 'clip'}{path.suffix}")
    return jsonify({"error": "Clip not found."}), 404


@app.delete("/api/clip/<clip_id>")
def api_clip_delete(clip_id: str):
    with gallery_lock:
        items = _read_gallery_unlocked()
        for item in items:
            if item.get("id") == clip_id:
                for _ in range(10):        # Windows: a player may still hold it
                    try:
                        (CLIPS_DIR / item["file"]).unlink(missing_ok=True)
                        break
                    except PermissionError:
                        time.sleep(0.2)
                    except OSError:
                        break
                shutil.rmtree(JOBS_DIR / item["id"], ignore_errors=True)
        bootstrap.atomic_write(GALLERY_PATH, json.dumps(
            [i for i in items if i.get("id") != clip_id], indent=2))
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# self-test: one real two-voice clip, checked
# --------------------------------------------------------------------------- #
SELFTEST_PATH = DATA_DIR / "selftest.json"
selftest_lock = threading.Lock()
selftest_run: selftest.Run | None = None
# The suite's stand-in engine has no GPU and placeholder weights; with this
# set, those two checks say "skipped (stand-in)" instead of failing. Never
# set it on a real install — the checks exist for the real card and files.
STANDIN = os.environ.get("MULTITALK_STUDIO_SELFTEST_STANDIN") == "1"


def example_dir() -> Path:
    """MultiTalk's own two-voice example, from the engine folder — or the
    repository's copy when the engine in use carries no examples."""
    for root in (bootstrap.engine_dir(cfg),
                 Path(bootstrap.DEFAULT_CONFIG["engine_dir"])):
        d = root.joinpath(*selftest.EXAMPLE)
        if all((d / f).is_file() for f in selftest.EXAMPLE_FILES.values()):
            return d
    raise FileNotFoundError("MultiTalk's examples/multi/1 is missing.")


def run_selftest(run: selftest.Run) -> None:
    try:
        _run_selftest(run)
    except Exception as exc:  # noqa: BLE001
        step = next((s["key"] for s in run.steps if s["state"] == "running"),
                    "engine")
        run.fail(step, f"{type(exc).__name__}: {exc}")
    finally:
        run.running = False
        try:
            bootstrap.atomic_write(SELFTEST_PATH, json.dumps(
                {**run.view(), "when": time.time()}, indent=2))
        except OSError:
            pass


def _run_selftest(run: selftest.Run) -> None:
    py, eng = bootstrap.engine_python(cfg), bootstrap.engine_dir(cfg)
    run.set("engine", "running", "Importing the engine…")
    if not py:
        return run.fail("engine", "The engine environment is not installed — "
                                  "run setup.")
    loads, why = bootstrap.check_engine_loads(py, eng)
    if not loads:
        return run.fail("engine", "The engine's code would not load: " + why)
    run.set("engine", "ok", "generate_multitalk.py imports and parses")

    run.set("gpu", "running", "Asking PyTorch…")
    vram, gpu = bootstrap.gpu_info(py, fresh=True)
    if vram:
        run.set("gpu", "ok", f"{gpu} · {vram / bootstrap.GIB:.0f} GB")
    elif STANDIN:
        run.set("gpu", "skipped", "Skipped — stand-in engine, no GPU here")
    else:
        return run.fail("gpu", "PyTorch cannot see an NVIDIA card. Update the "
                               "NVIDIA driver, then on this page press "
                               "Reinstall next to PyTorch.")

    run.set("weights", "running", "Measuring the files…")
    whole, detail = selftest.weights_whole(bootstrap.weights_dir(cfg), cfg)
    if not whole and not STANDIN:
        return run.fail("weights", detail)
    run.set("weights", "ok" if whole else "skipped",
            detail if whole else "Skipped — stand-in weights")

    run.set("render", "running", "Queued…")
    ex = example_dir()
    names = {}
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for key, fname in selftest.EXAMPLE_FILES.items():
        name = uuid.uuid4().hex[:12] + Path(fname).suffix.lower()
        shutil.copyfile(ex / fname, UPLOADS_DIR / name)
        names[key] = name
    defaults = bootstrap.PRECISIONS.get(cfg.get("precision") or "",
                                        bootstrap.PRECISIONS["int8-fusionx"])
    settings = engine.normalize({
        **names, "people": 2, "source": "files", "audio_type": "add",
        "mode": "clip", "size": "multitalk-240", "seed": 42,
        "output": engine.DEFAULT_OUTPUT,
        "steps": min(defaults["defaults"]["steps"], 8),
        "prompt": "A man and a woman sit at a table and talk to each other, "
                  "taking turns.",
        "title": "Engine self-test"}, cfg)
    blocker = engine_blocker()
    if blocker:
        return run.fail("render", blocker)
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "pct": 0,
                        "stage": "Waiting for the GPU", "created": time.time(),
                        "title": "Engine self-test", "settings": settings,
                        "log": [], "proc": None}
    run.job_id = job_id
    wake.set()
    while True:
        with jobs_lock:
            job = dict(jobs.get(job_id) or {})
        if job.get("status") in ("done", "error", "cancelled"):
            break
        run.set("render", "running", job.get("stage") or "Working…")
        time.sleep(1)
    if job["status"] != "done":
        return run.fail("render", job.get("error") or "The test render was "
                                                      "stopped.")
    run.took = round(job["finished"] - job["started"], 1)
    run.clip = job["clip"]["id"]
    run.set("render", "ok", f"{run.took:.0f} s for 3.2 s of video at 320 px, "
                            f"{settings['steps']} steps")

    run.set("frames", "running", "Looking at the frames…")
    ffmpeg = selftest.find_ffmpeg(py)
    if not ffmpeg:
        return run.fail("frames", "No ffmpeg to read the video with — install "
                                  "the engine packages on this page.")
    video = CLIPS_DIR / job["clip"]["file"]
    want = engine.OUTPUTS[engine.DEFAULT_OUTPUT]
    got = selftest.video_size(ffmpeg, video)
    if got != (want["w"], want["h"]):
        return run.fail("frames", f"The video is {got[0]} × {got[1]}, not "
                                  f"{want['w']} × {want['h']}.")
    ok, detail = selftest.judge_frames(
        selftest.frame_stats(selftest.read_frames(ffmpeg, video)),
        engine.FRAME_NUM)
    if not ok:
        return run.fail("frames", detail)
    run.set("frames", "ok", f"{got[0]} × {got[1]} · " + detail)

    run.set("voices", "running", "Listening…")
    rate = 8000
    turns = [selftest.wav_seconds(ex / selftest.EXAMPLE_FILES["audio1"]),
             selftest.wav_seconds(ex / selftest.EXAMPLE_FILES["audio2"])]
    ok, detail = selftest.judge_voices(
        selftest.read_audio(ffmpeg, video, rate), rate, turns,
        engine.FRAME_NUM / engine.FPS)
    if not ok:
        return run.fail("voices", detail)
    run.set("voices", "ok", detail)
    run.estimate = selftest.estimate(run.took, engine.clips_for(250))
    run.ok = True


@app.post("/api/selftest")
def api_selftest_start():
    global selftest_run
    with selftest_lock:
        if selftest_run and selftest_run.running:
            return jsonify({"error": "The self-test is already running."}), 409
        selftest_run = selftest.Run()
        threading.Thread(target=run_selftest, args=(selftest_run,),
                         daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/selftest")
def api_selftest_state():
    if selftest_run:
        return jsonify(selftest_run.view())
    last = None
    if SELFTEST_PATH.exists():
        try:
            last = json.loads(SELFTEST_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            last = None
    return jsonify(last or selftest.Run().view() | {"running": False})


def _warm_gpu() -> None:
    """Read the GPU once in the background so the page can offer settings
    that fit the card without waiting on a torch import."""
    python = bootstrap.engine_python(cfg)
    if python:
        bootstrap.gpu_info(python)


def boot() -> None:
    """Verify every saved location (searching the drives if the engine is
    lost), then read the GPU."""
    _heal(search=True)
    _warm_gpu()


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    for d in (DATA_DIR, CLIPS_DIR, UPLOADS_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    # the page can open while the saved locations are checked
    threading.Thread(target=boot, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  MultiTalk Studio  ->  {url}\n")
    if os.environ.get("MULTITALK_STUDIO_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
