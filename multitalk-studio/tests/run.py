#!/usr/bin/env python3
"""Run the suite.

    python tests/run.py              everything available
    python tests/run.py units api    just those
    python tests/run.py --list       what there is
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

MODULES = [
    ("gate", "compile, parse, ids and wiring — the checks after any edit"),
    ("units", "the pure logic, against the real engine's own code"),
    ("reuse", "offline model reuse and one-shot location recovery"),
    ("api", "the HTTP surface, end to end against a fake engine"),
    ("ui", "the interface itself, in a browser"),
]


def gate() -> tuple[int, list[str]]:
    passed, failures = 0, []
    sources = ["server.py", "bootstrap.py", "manager.py", "engine.py"]
    done = subprocess.run([sys.executable, "-m", "py_compile", *sources],
                          cwd=ROOT, capture_output=True, text=True)
    if done.returncode == 0:
        passed += 1
        print(f"  ok   {' '.join(sources)} all compile")
    else:
        failures.append("python compile")
        print(f"  FAIL python compile — {done.stderr.strip()[:200]}")

    page = (ROOT / "web" / "index.html").read_text()
    script = "\n".join(re.findall(r"<script>(.*?)</script>", page, re.S))
    scratch = HERE / ".inline.js"
    scratch.write_text(script)
    try:
        done = subprocess.run(["node", "--check", str(scratch)],
                              capture_output=True, text=True)
        if done.returncode == 0:
            passed += 1
            print("  ok   the inline script parses")
        else:
            failures.append("inline script")
            print(f"  FAIL inline script — {done.stderr.strip()[:300]}")
    except FileNotFoundError:
        print("  --   node is not installed, so the inline script was not checked")
    finally:
        scratch.unlink(missing_ok=True)

    markup = page[:page.index("<script>")]
    defined = set(re.findall(r'\bid="([^"]+)"', markup))
    missing = sorted({u for u in re.findall(r'\$\("([^"]+)"\)', script)
                      if u not in defined})
    if missing:
        failures.append("missing ids")
        print(f"  FAIL the script reaches for ids that do not exist: {missing}")
    else:
        passed += 1
        print("  ok   every id the script reaches for exists")

    def wired(control: str) -> bool:
        for m in re.finditer(re.escape(f'$("{control}")'), script):
            if "addEventListener" in script[m.end():m.end() + 200]:
                return True
        for m in re.finditer(r'\[(?:\s*\[?"[^"]+"[^\]]*\]?,?\s*)+\]\.forEach',
                             script):
            if f'"{control}"' in m.group(0) and \
                    "addEventListener" in script[m.end():m.end() + 500]:
                return True
        return False

    interactive = set(re.findall(r'<button[^>]*\bid="([^"]+)"', markup))
    interactive |= set(re.findall(r'<div class="chips" id="([^"]+)"', markup))
    interactive |= set(re.findall(r'<input type="(?:range|file)" id="([^"]+)"',
                                  markup))
    deaf = sorted(c for c in interactive if not wired(c))
    if deaf:
        failures.append("deaf controls")
        print(f"  FAIL controls no listener ever touches: {deaf}")
    else:
        passed += 1
        print("  ok   every interactive control has a listener")

    done = subprocess.run(["bash", "-n", str(ROOT / "run.sh")],
                          capture_output=True, text=True)
    if done.returncode == 0:
        passed += 1
        print("  ok   run.sh parses")
    else:
        failures.append("run.sh")
        print(f"  FAIL run.sh — {done.stderr.strip()[:200]}")
    if b"\r\n" in (ROOT / "run.bat").read_bytes():
        passed += 1
        print("  ok   run.bat has Windows line endings")
    else:
        failures.append("run.bat line endings")
        print("  FAIL run.bat needs CRLF line endings or cmd.exe misreads it")
    return passed, failures


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        for name, what in MODULES:
            print(f"  {name:8s} {what}")
        return 0
    chosen = [m for m in MODULES if not args or m[0] in args]
    results, started = [], time.time()
    for name, what in chosen:
        print(f"\n=== {name} — {what} ===")
        began = time.time()
        if name == "gate":
            passed, failures = gate()
        else:
            module = importlib.import_module(f"test_{name}")
            skip = getattr(module, "available", lambda: "")()
            if skip:
                print(f"  --   skipped: {skip}")
                results.append((name, 0, [], 0.0, skip))
                continue
            try:
                suite = module.run()
                passed, failures = suite.passed, suite.failures
            except Exception as exc:  # noqa: BLE001
                import traceback

                import harness
                part = harness.CURRENT
                passed = part.passed if part else 0
                failures = (list(part.failures) if part else []) + \
                    [f"{type(exc).__name__}: {exc}"]
                print("       " + traceback.format_exc().strip()
                      .replace("\n", "\n       ")[-1500:])
        results.append((name, passed, failures, time.time() - began, ""))
    print("\n" + "=" * 62)
    for name, passed, failures, took, skipped in results:
        if skipped:
            print(f"  {name:8s} skipped — {skipped}")
        else:
            state = "ok" if not failures else f"{len(failures)} FAILED"
            print(f"  {name:8s} {passed:3d} passed  {state:>10s}  {took:5.1f}s")
    print("=" * 62)
    broken = [r for r in results if r[2]]
    for name, _, failures, _, _ in broken:
        for f in failures:
            print(f"    {name}: {f}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
