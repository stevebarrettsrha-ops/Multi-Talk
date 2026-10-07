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

## Validation gate

```bash
python tests/run.py          # gate + units + api + ui, ~15 s
```
