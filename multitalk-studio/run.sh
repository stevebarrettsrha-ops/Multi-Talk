#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it, then run this again." >&2
  exit 1
fi
# The studio's own small venv (Flask, requests). Current Debian/Ubuntu refuse
# pip into the system Python (PEP 668). A venv without pip is rebuilt.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
  "$PY" -m venv .venv || {
    rm -rf .venv
    echo "Could not create a virtual environment. On Debian/Ubuntu install" >&2
    echo "python3-venv (sudo apt install python3-venv), then run this again." >&2
    exit 1
  }
fi
PY=.venv/bin/python
echo "  Using: $($PY -c 'import sys;print(sys.executable)')"
"$PY" -m pip install --disable-pip-version-check --quiet -r requirements.txt
exec "$PY" server.py
