# MultiTalk Studio — invariants

## Hard rules

1. **`web/index.html` stays one file, no build step.** Same shell as the
   sibling studios: rail, prompt bar over a masonry feed, popover settings,
   Models and Engine pages, setup sheet.
2. **The engine runs out of process.** Each render is one run of
   `../MultiTalk/generate_multitalk.py` in `engine-venv`, started by
   `server.run_job` through `bootstrap.stream`. Never import torch or the
   model into `server.py`: an out-of-memory stop must cost one job, not the
   app. One render at a time — the queue is the GPU's.
3. **The command line is built only in `engine.argv()`**, and every flag
   must exist in the real parser. `tests/test_units.py` reads
   `generate_multitalk.py` to check this; the fake engine's parser mirrors
   it. Add a flag to the engine first, then to `argv()`.
4. **Paths in the input JSON use forward slashes** (`engine.as_path`).
   `generate_multitalk.py` names its audio folder from
   `cond_image.split('/')[-1]`; a Windows backslash path becomes the whole
   path.
5. **The 8 GB defaults are the default.** INT8, FusionX, DiT fully offloaded
   (`--num_persistent_param_in_dit 0`), `--t5_cpu`, `--vae_tile 32`,
   `multitalk-360`. `bootstrap.recommended()` loosens them only for cards
   that measured larger.
6. **The preflight tells the truth.** `assess()` is a pure function with unit
   tests. 8 GB VRAM with 32 GB RAM is "tight", 16 GB RAM is "hard" (the INT8
   weights live in RAM), under ~6 GB VRAM is "hard". The README says plainly
   that no real RTX 4060 timing exists yet; keep it that way until one does.
7. **Downloads are resumable** (`.part`, `Range`, 64 kB chunks so a dropped
   connection keeps what arrived), **deletes are path-checked**
   (`manager.delete_model`: relative, no `..`, under the weights folder),
   **uploads are renamed** to `<12 hex>.<ext>` and only served by that name.
8. **Localhost only.** `local_only()` refuses foreign Host headers and
   cross-origin writes — the app runs pip and processes.

## Two output sizes, exact

A finished clip is exactly 720 × 360 or 720 × 1280 (`engine.OUTPUTS`;
default 720 × 360), and nothing else is accepted. The engine renders the
bucket of that shape (`--bucket_ratio`, height / width) at the chosen
"render at" size, then `server.run_job` resizes it with
`engine.resize_args` (fill, centre-crop, `-c:a copy`) and replaces the
render. The API test measures both sizes with ffmpeg. Do not offer a size
the 8 GB card cannot hold as a native render: the resize is the route.

## Every running download is visible

A person downloading 29 GB must always see it move. Each Models-page file
row shows its own bar while it downloads (`paintModelRows`, matched to tasks
by repo and path, folder rows summed), the set shows one overall bar, the
Downloads panel lists every running task before any finished one, and
`Tasks.visible()` never drops a running task from `/api/tasks`. The Engine
page's Activity panel has a bar too. pip's raw progress is detected by
letting pip parse `--progress-bar raw` (24.1+), and every line it prints is
passed on. `test_ui` fails if the biggest file downloads without a moving
bar.

## The weight set

`bootstrap.model_set()` is the single list. INT8 only: the files the engine
opens by name in quant mode are `quant_models/{dit_model,t5}_int8.*` (base)
or `quant_model_int8_FusionX.safetensors` + `quantization_map_int8_FusionX.json`
(FusionX, passed as `--lora_dir`). The 65 GB bf16 shards and the bf16 T5 are
never needed in quant mode and are not downloaded. Folder items (`prefix`)
are expanded from the repo tree at download time. wav2vec's safetensors come
from `refs/pr/1`, as upstream's README does it.

## Engine patches

Every change in `../MultiTalk` is marked "MultiTalk Studio" and listed in the
README. `../MultiTalk/tests/test_lowvram_patches.py` checks the tiled VAE
against the plain one, the SDPA fallbacks against plain softmax attention,
and the new buckets, on CPU with random weights.

## Setup's last step imports the engine

`bootstrap.check_engine_loads()` runs `generate_multitalk.py --help` in the
engine environment. Finding the packages is not enough: the first real
run-through found upstream's `from inspect import ArgSpec`, which crashes on
Python 3.11+, and only an import shows that. Keep the engine importable
without a GPU (no CUDA calls at import time) so this check works anywhere.

## The self-test is the answer to "does it work"

`selftest.py` + `server.run_selftest`: one 3-second two-voice clip from
`MultiTalk/examples/multi/1`, through the normal render queue, then checks
on what came back (frames with contrast that move, each voice audible in its
turn, weights at least half their published size). The judgements are pure
functions with unit tests. `MULTITALK_STUDIO_SELFTEST_STANDIN=1` exists only
for the suite: it turns the GPU and weight-size steps into "skipped" — never
"ok" — because the stand-in has neither. Never set it on a real install.

## The tiny engine runs the real pipeline on a CPU

`tests/tiny_engine/generate_multitalk.py` swaps `wan.MultiTalkPipeline` for
a subclass whose loading builds a tiny random DiT, the full-size Wan VAE with
random weights, and stubs for umT5/CLIP of the real output shapes; then calls
the real `generate()`. It runs in float32 (upstream leans on CUDA autocast
to mix bf16 and float32, which does nothing for CPU tensors) and hides
xformers (no CPU kernels). It is how the `--t5_cpu` crash was found: use it
after engine changes, with `make_weights.py` for the weights folder.

## Saved locations are verified, never trusted

The config keeps absolute paths (`engine_dir` — even the default is saved
absolute — `weights_dir`, `python`), which go stale the moment the repo is
moved, renamed or re-extracted, and then everything reads "missing" though it
is all on disk. `bootstrap.verify_locations()` reuses persisted successful checks and failed
search attempts. Import, boot and `/api/deps` perform cheap path-health checks;
a changed/missing path gets one recovery attempt, while only the explicit
Recheck button (`/api/deps?fresh=1&relocate=1`) forces another attempt for an
unchanged failure. `fresh=1` by itself is a package refresh, not relocation.
`heal_paths()` grafts a stale path's tail onto the repo's current folder
(`rebase_path`, longest tail first, so a renamed root works) and clears an
engine Python that is gone. If the engine is still nowhere,
`find_engine_installs()` walks the drives breadth-first under a time budget
and `pick_engine()` prefers the checkout holding the weights. While it walks,
`/api/deps` reports `searching` and the page re-polls.
`MULTITALK_STUDIO_NO_SEARCH=1` (set by the test harness) turns it off: test
configs name made-up folders on purpose.

## Validation gate

```bash
python tests/run.py          # gate + units + api + ui, ~15 s
```
