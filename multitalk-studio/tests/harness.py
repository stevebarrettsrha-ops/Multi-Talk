"""Shared plumbing for the suite: reporting, and disposable servers.

Every test gets its own port and its own data directory. Nothing here touches
the gallery, config or weights of a real install.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAKE_ENGINE = HERE / "fake_engine"
SAMPLE_PNG = HERE / "sample.png"
SAMPLE_WAV = HERE / "sample.wav"
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
sys.path.insert(0, str(ROOT))

CURRENT: "Suite | None" = None


class Suite:
    """Collects checks so one failure does not hide the rest."""

    def __init__(self, name: str) -> None:
        global CURRENT
        self.name, self.passed, self.failures = name, 0, []
        CURRENT = self

    def check(self, what: str, ok, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  ok   {what}" + (f" — {detail}" if detail else ""))
        else:
            self.failures.append(what)
            print(f"  FAIL {what}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    def equal(self, what: str, got, want) -> bool:
        return self.check(what, got == want, f"got {got!r}, wanted {want!r}")

    def fails_with(self, what: str, fn, expect: type = Exception,
                   contains: str = "") -> bool:
        try:
            fn()
        except expect as exc:
            return self.check(what, contains.lower() in str(exc).lower(),
                              f"said {str(exc)[:80]!r}")
        except Exception as exc:  # noqa: BLE001
            return self.check(what, False, f"raised {type(exc).__name__}: {exc}")
        return self.check(what, False, "did not raise at all")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """A subprocess that is always cleaned up, however the test ends."""

    def __init__(self, argv, port, ready_path, env=None, cwd=ROOT):
        self.argv, self.port, self.ready_path = argv, port, ready_path
        self.env, self.cwd = env or {}, cwd
        self.proc = None
        self.url = f"http://127.0.0.1:{port}"
        self.log = Path(tempfile.mkstemp(suffix=".log")[1])

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.argv, cwd=str(self.cwd), env={**os.environ, **self.env},
            stdout=self.log.open("w"), stderr=subprocess.STDOUT,
            start_new_session=True)
        for _ in range(120):
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.argv[-1]} died at startup:\n"
                                   f"{self.log.read_text()[-2000:]}")
            try:
                requests.get(self.url + self.ready_path, timeout=2)
                return self
            except Exception:
                time.sleep(0.25)
        raise RuntimeError(f"{self.argv[-1]} never answered:\n"
                           f"{self.log.read_text()[-2000:]}")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def tail(self, n: int = 30) -> str:
        return "\n".join(self.log.read_text().splitlines()[-n:])


class Workspace:
    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="mt-test-"))
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def fake_weights(wdir: Path, precision: str = "int8-fusionx",
                 tts: bool = True) -> None:
    """Put a file wherever model_set() looks, folder items included."""
    import bootstrap
    for item in bootstrap.model_set({"precision": precision, "want_tts": tts}):
        target = bootstrap.model_path(wdir, item)
        if item["prefix"]:
            target.mkdir(parents=True, exist_ok=True)
            names = ["af_heart.pt", "am_adam.pt"] if "voices" in item["path"] \
                else ["tokenizer.json"]
            for n in names:
                (target / n).write_bytes(b"\x00" * 8)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * 16)


def studio(data: Path, weights: Path, engine_dir: Path = FAKE_ENGINE,
           env: dict | None = None, **config) -> Server:
    data.mkdir(parents=True, exist_ok=True)
    (data / "config.json").write_text(json.dumps({
        "engine_dir": str(engine_dir), "weights_dir": str(weights),
        "python": sys.executable, "precision": "int8-fusionx",
        "want_tts": True, "setup_complete": True, **config}))
    port = free_port()
    return Server([sys.executable, "server.py"], port, "/api/status",
                  env={"MULTITALK_STUDIO_PORT": str(port),
                       "MULTITALK_STUDIO_NO_BROWSER": "1",
                       "MULTITALK_STUDIO_NO_SEARCH": "1",
                       "MULTITALK_STUDIO_DATA": str(data), **(env or {})})


def hub() -> Server:
    port = free_port()
    return Server([sys.executable, str(HERE / "mock_hf.py"), str(port)],
                  port, "/mock/log")


def upload(url: str, path: Path, name: str | None = None) -> dict:
    with open(path, "rb") as fh:
        r = requests.post(url + "/api/upload",
                          files={"file": (name or path.name, fh)}, timeout=30)
    r.raise_for_status()
    return r.json()


def wait_for(cond, timeout: float = 30, step: float = 0.3) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(step)
    return False


def finish(url: str, timeout: float = 60) -> list[dict]:
    wait_for(lambda: not [j for j in requests.get(url + "/api/jobs",
                                                  timeout=10).json()
                          if j["status"] in ("running", "queued")], timeout)
    return requests.get(url + "/api/jobs", timeout=10).json()
