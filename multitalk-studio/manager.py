"""
manager.py - getting the machine ready, driven from the front end.

Dependencies: Python, the engine source, the engine environment, PyTorch,
the engine packages and the weights — each with an installer. HuggingFace:
download the set or single files, resume, cancel, delete.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

import bootstrap


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
class Task:
    def __init__(self, kind: str, title: str, meta: dict | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.meta = meta or {}
        self.state = "running"
        self.pct = 0.0
        self.got = 0          # bytes so far and expected, for downloads
        self.total = 0
        self.busy = False     # working with no number to show (pip unpacking)
        self.detail = ""
        self.lines: list[str] = []
        self.created = time.time()
        self.cancel = False
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 1200:
                del self.lines[:600]

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def view(self, since: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "title": self.title,
                    "meta": self.meta, "state": self.state,
                    "got": self.got, "total": self.total, "busy": self.busy,
                    "pct": round(self.pct or 0, 1), "detail": self.detail,
                    "created": self.created, "cursor": len(self.lines),
                    "lines": self.lines[since:]}


class Tasks:
    def __init__(self) -> None:
        self._items: dict[str, Task] = {}
        self._lock = threading.Lock()

    def add(self, task: Task) -> Task:
        with self._lock:
            self._items[task.id] = task
            finished = sorted((t for t in self._items.values()
                               if t.state != "running"), key=lambda t: t.created)
            for old in finished[:-40]:
                self._items.pop(old.id, None)
        return task

    def get(self, task_id: str) -> Task | None:
        return self._items.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            return sorted(self._items.values(), key=lambda t: t.created,
                          reverse=True)

    def visible(self, finished: int = 25) -> list[Task]:
        """What the page lists: every running task, then the newest finished
        ones. A plain newest-25 cap hid the oldest running downloads — which
        are the big ones, since the set queues them first."""
        tasks = self.list()
        return [t for t in tasks if t.state == "running"] + \
            [t for t in tasks if t.state != "running"][:finished]

    def running(self, kind: str = "") -> list[Task]:
        return [t for t in self.list()
                if t.state == "running" and (not kind or t.kind == kind)]


TASKS = Tasks()


def spawn(kind: str, title: str, fn, meta: dict | None = None) -> Task:
    task = TASKS.add(Task(kind, title, meta))

    def wrapper():
        try:
            fn(task)
            if task.state == "running":
                task.set(state="done", pct=100)
        except Exception as exc:  # noqa: BLE001
            if task.cancel:
                task.set(state="cancelled", detail="Cancelled")
                return
            task.log(f"FAILED: {exc}")
            task.set(state="error", detail=str(exc))

    threading.Thread(target=wrapper, daemon=True).start()
    return task


# --------------------------------------------------------------------------- #
# dependency report
# --------------------------------------------------------------------------- #
_CHECK_SEEN: dict[str, dict] = {}


def engine_report(python: str, fresh: bool = False) -> dict:
    """check_engine() is a 5-15 s torch import; keep a good answer until
    something is installed."""
    if python in _CHECK_SEEN and not fresh:
        return _CHECK_SEEN[python]
    report = bootstrap.check_engine(python)
    if report.get("ok"):
        _CHECK_SEEN[python] = report
    return report


def forget() -> None:
    _CHECK_SEEN.clear()
    bootstrap.forget_gpu()


def dependencies(cfg: dict, searching: bool = False) -> list[dict]:
    items: list[dict] = []
    try:
        py = bootstrap.find_python()
        items.append({"id": "python", "label": "Python 3.10–3.12",
                      "state": "ok", "detail": py, "action": None})
    except Exception as exc:  # noqa: BLE001
        items.append({"id": "python", "label": "Python 3.10–3.12",
                      "state": "missing", "detail": str(exc), "action": None})

    eng = bootstrap.engine_dir(cfg)
    have_src = (eng / "generate_multitalk.py").exists()
    if have_src:
        items.append({"id": "engine", "label": "MultiTalk engine",
                      "state": "ok", "detail": str(eng), "action": None})
    elif searching:
        items.append({"id": "engine", "label": "MultiTalk engine",
                      "state": "warn",
                      "detail": "Searching this computer for it — this list "
                                "updates when the search is done.",
                      "action": None})
    else:
        items.append({"id": "engine", "label": "MultiTalk engine",
                      "state": "missing",
                      "detail": f"Not found at {eng}. Point Settings at "
                                "the MultiTalk folder.",
                      "action": None})

    vpy = bootstrap.engine_python(cfg)
    items.append({"id": "venv", "label": "Engine environment",
                  "state": "ok" if vpy else "missing",
                  "detail": vpy or "A private Python environment for the "
                                   "engine, so nothing touches your system "
                                   "Python.",
                  "action": "reinstall" if vpy else "install"})

    if vpy:
        report = engine_report(vpy)
        torch_v = report.get("torch", "")
        if not torch_v:
            items.append({"id": "torch", "label": "PyTorch", "state": "missing",
                          "detail": report.get("error") or "Not installed.",
                          "action": "install"})
        else:
            vram, gpu = bootstrap.gpu_info(vpy)
            if report.get("cuda"):
                gb = vram / bootstrap.GIB
                items.append({"id": "torch", "label": "PyTorch",
                              "state": "ok" if gb >= 7 else "warn",
                              "detail": f"torch {torch_v} — {gpu}, {gb:.0f} GB"
                                        + (" · xformers" if report.get("xformers")
                                           else ""),
                              "action": "reinstall"})
            else:
                items.append({"id": "torch", "label": "PyTorch",
                              "state": "warn",
                              "detail": f"torch {torch_v} — no NVIDIA GPU "
                                        "found. MultiTalk needs one.",
                              "action": "reinstall"})
        missing = report.get("missing")
        if missing is None and not torch_v:
            items.append({"id": "packages", "label": "Engine packages",
                          "state": "unknown", "detail": "Install PyTorch first.",
                          "action": "install"})
        else:
            missing = [m for m in (missing or []) if m not in ("torch",
                                                               "torchvision")]
            items.append({"id": "packages", "label": "Engine packages",
                          "state": "missing" if missing else "ok",
                          "detail": ("Missing: " + ", ".join(missing))
                          if missing else "transformers, diffusers, "
                          "optimum-quanto, librosa, misaki and the rest",
                          "action": "install" if missing else "update"})
    else:
        for dep_id, label in (("torch", "PyTorch"),
                              ("packages", "Engine packages")):
            items.append({"id": dep_id, "label": label, "state": "unknown",
                          "detail": "Create the engine environment first.",
                          "action": None})

    ffmpeg = shutil.which("ffmpeg")
    items.append({"id": "ffmpeg", "label": "ffmpeg", "state": "ok",
                  "detail": ffmpeg or "Using the copy imageio-ffmpeg ships "
                                      "with (installed with the packages).",
                  "action": None})

    wdir = bootstrap.weights_dir(cfg)
    missing = bootstrap.missing_models(wdir, cfg)
    items.append({"id": "models", "label": "MultiTalk weights",
                  "state": "missing" if missing else "ok",
                  "detail": ("Missing: " + ", ".join(m["name"] for m in missing))
                  if missing else f"All present in {wdir}",
                  "action": "models"})
    return items


# --------------------------------------------------------------------------- #
# installers
# --------------------------------------------------------------------------- #
def install_dependency(dep_id: str, cfg: dict) -> Task:
    titles = {"venv": "Create the engine environment",
              "torch": "Install PyTorch",
              "packages": "Install the engine packages"}
    if dep_id not in titles:
        raise RuntimeError(f"Nothing to install for '{dep_id}'.")
    if TASKS.running("dependency"):
        raise RuntimeError("Another install is running — wait for it.")

    def run(task: Task) -> None:
        def pct(p, d):
            # None means pip is unpacking or building: keep the bar where it
            # is and mark it busy, rather than snapping it back to zero
            if p is None:
                task.set(busy=True, detail=d)
            else:
                task.set(pct=p, busy=False, detail=d)

        try:
            if dep_id == "venv":
                base = bootstrap.find_python()
                vpy = bootstrap.make_venv(base, task.log)
                cfg["python"] = str(vpy)
                bootstrap.save_config(cfg)
                bootstrap.pip_install(str(vpy), ["--upgrade", "pip", "wheel",
                                                 "setuptools"], task.log)
                task.set(detail=str(vpy))
                return
            vpy = bootstrap.engine_python(cfg)
            if not vpy:
                raise RuntimeError("Create the engine environment first.")
            if dep_id == "torch":
                task.set(detail="Installing PyTorch — the long one…")
                bootstrap.install_torch(cfg, vpy, task.log, pct,
                                        lambda: task.cancel)
            else:
                task.set(detail="Installing the engine packages…")
                bootstrap.install_packages(vpy, task.log, pct,
                                           lambda: task.cancel)
            task.set(detail="Installed.")
        finally:
            forget()

    return spawn("dependency", titles[dep_id], run, {"dep": dep_id})


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def download_item(cfg: dict, item: dict) -> Task:
    wdir = bootstrap.weights_dir(cfg)
    dest = bootstrap.model_path(wdir, item)
    if any(t.meta.get("dest") == str(dest) for t in TASKS.running("download")):
        raise RuntimeError(f"{item['name']} is already downloading.")

    def run(task: Task) -> None:
        task.log(f"{item['repo']}/{item['path']} → {dest}")

        def on_prog(got, total, speed, eta):
            task.set(pct=(got / total * 100) if total else 0, got=got,
                     total=total, busy=not total,
                     detail=bootstrap.fmt_transfer(got, total, speed, eta))

        bootstrap.download_file(cfg, item["repo"], item["path"], dest, on_prog,
                                lambda: task.cancel, item.get("revision", "main"),
                                expected_size=item.get("known_size", 0))
        if task.cancel:
            task.set(state="cancelled",
                     detail="Cancelled — the part that downloaded is kept, and "
                            "starting again carries on from there.")
            return
        size = dest.stat().st_size
        task.set(pct=100, got=size, total=size, busy=False,
                 detail=f"Saved — {bootstrap.fmt_size(size)}")

    return spawn("download", item["name"], run,
                 {"repo": item["repo"], "path": item["path"], "dest": str(dest),
                  "size": item.get("size") or 0})


def download_set(cfg: dict) -> list[Task]:
    """Queue whatever the chosen precision still needs, folder items
    expanded into their files."""
    wdir = bootstrap.weights_dir(cfg)
    todo = [m for m in bootstrap.model_set(cfg) if not bootstrap.present(wdir, m)]
    if not todo:
        return []
    files = [f for f in bootstrap.expand(cfg, todo)
             if not bootstrap.finished_file(bootstrap.model_path(wdir, f),
                                            f.get("known_size", 0))]
    busy = {t.meta.get("dest") for t in TASKS.running("download")}
    return [download_item(cfg, f) for f in files
            if str(bootstrap.model_path(wdir, f)) not in busy]


def curated(cfg: dict) -> dict:
    wdir = bootstrap.weights_dir(cfg)
    sets = {}
    for key, p in bootstrap.PRECISIONS.items():
        probe = dict(cfg, precision=key)
        files = [{**m, "installed": bootstrap.present(wdir, m)}
                 for m in bootstrap.model_set(probe)]
        sets[key] = {"label": p["label"], "note": p["note"], "files": files,
                     "complete": all(f["installed"] for f in files
                                     if f["role"] == "required"),
                     "download": sum(f["size"] for f in files)}
    return {"precision": cfg.get("precision"), "sets": sets,
            "weights_dir": str(wdir)}


def local_models(cfg: dict) -> list[dict]:
    wdir = bootstrap.weights_dir(cfg)
    out: list[dict] = []
    if not wdir.is_dir():
        return out
    for f in sorted(wdir.rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".pth", ".pt", ".bin",
                                        ".json", ".part") and f.stat().st_size:
            rel = f.relative_to(wdir).as_posix()
            if rel.count("/") > 3 or "/voices/" in rel:
                continue
            out.append({"path": rel, "size": f.stat().st_size,
                        "partial": f.suffix == ".part"})
    return out


def delete_model(cfg: dict, rel: str) -> None:
    """Only a file under the weights folder, named relative to it."""
    if not rel or "\\" in rel or ".." in rel.split("/") or rel.startswith("/"):
        raise RuntimeError("That path is not allowed.")
    root = bootstrap.weights_dir(cfg).resolve()
    target = (root / rel).resolve()
    if target == root or not target.is_relative_to(root):
        raise RuntimeError("That path is outside the weights folder.")
    if not target.is_file():
        raise RuntimeError("That file is already gone.")
    target.unlink()


def voices(cfg: dict) -> list[str]:
    vdir = bootstrap.weights_dir(cfg) / bootstrap.KOKORO_DIR / "voices"
    if not vdir.is_dir():
        return []
    return sorted(f.stem for f in vdir.glob("*.pt"))
