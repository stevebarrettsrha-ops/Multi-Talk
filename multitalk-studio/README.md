# MultiTalk Studio

A local front end for **MeiGen MultiTalk**. You give it a picture of one or
two people and what they say, as a recording or typed text. It gives back a
video of them saying it, lips in sync. It follows the same pattern as the
other studios in this account: a small Flask app, one HTML page, a setup
sheet, and Models and Engine pages.

The engine is the MultiTalk code in `../MultiTalk`, patched to fit a small
card (see [What changed in the engine](#what-changed-in-the-engine)). The
studio runs its own `generate_multitalk.py` in a private Python
environment, one render at a time, and reads its output for the progress
bar.

---

## Read this before you download 29 GB

MultiTalk is a 14-billion-parameter video model. In bf16 it wants about 30
GB of VRAM. This app runs it on an **RTX 4060 (8 GB)** by:

- **Using MeiGen's INT8 build.** The DiT and the text encoder are about half
  their bf16 size.
- **Streaming the model from RAM.** The DiT stays in system RAM and visits
  the GPU one layer at a time (`num_persistent_param_in_dit 0`).
- **Encoding the prompt on the CPU.** The 6.7 GB text encoder never touches
  the GPU.
- **Decoding in tiles.** The VAE decodes 256 px tiles one at a time
  instead of the whole frame.
- **Rendering at 480 px** instead of the 640 px it was trained at.
- **Using FusionX.** The default model has the FusionX acceleration merged
  in, so it needs 8 sampling steps instead of 40.

The weights are the floor:

| What | Size |
|---|---|
| INT8 MultiTalk DiT (FusionX or base) | ≈ 16.5 GB |
| INT8 umT5-XXL text encoder | ≈ 6.7 GB |
| Wan2.1 CLIP image encoder | 4.8 GB |
| Wan2.1 VAE, tokenizers, wav2vec2 | ≈ 0.9 GB |
| Kokoro text-to-speech (optional) | ≈ 0.4 GB |

The ≈ sizes are estimates until HuggingFace is asked. Setup reads the real
ones before it starts.

**System RAM matters as much as VRAM.** The INT8 model and text encoder live
in RAM while the GPU borrows them, so a render peaks around 29 GB of RAM.
32 GB is the practical minimum. With 16 GB, Windows pages to disk for most
of every step.

**Expect minutes per clip, not seconds.** The model renders in 3.2 second
clips: 81 frames at 25 fps. Longer speech is a chain of clips, each adding
2.2 seconds. On an RTX 4060 at 480 px with FusionX, budget several minutes
per clip. Ten seconds of speech is five clips. Nothing here has been timed
on a real RTX 4060 yet. The first real render will give a true number.

The **Preflight** panel on the Engine page measures your VRAM, RAM and free
disk before anything downloads and says which of those applies.

---

## Running it

**Windows**: double-click `run.bat`. **Linux and macOS**: run `./run.sh`.
Both open <http://127.0.0.1:7806>.

The studio needs Python 3.10 or newer. The engine needs **Python 3.10, 3.11
or 3.12**, because PyTorch 2.4 publishes no wheels for 3.13. If you only have
3.13, install 3.12 from python.org alongside it.

The first launch opens the setup sheet. It runs these steps:

1. Finds a suitable Python.
2. Creates `engine-venv/`, a private environment for the engine.
3. Installs PyTorch 2.4.1 (CUDA 12.1) and, optionally, xformers.
4. Installs the engine's packages from `engine-requirements.txt`.
5. Downloads the weights into `../MultiTalk/weights/`, resumably.
6. Imports the real engine in the new environment and runs its argument
   parser, so a package or code problem shows up here rather than at your
   first render.

Each step shows live progress. A failed or cancelled download keeps what
arrived and resumes next time.

flash-attn and xfuser are **not** installed. flash-attn has no Windows
wheels and xfuser is only for multi-GPU runs. The engine falls back to
PyTorch's own fused attention without them.

---

## Making a clip

1. **Picture.** A photo or drawing of the people. Cartoons work. The output
   keeps the picture's aspect ratio.
2. **People.** One or two. With two, voice 1 drives the person on the
   **left** of the picture and voice 2 the **right**.
3. **Speech.**
   - **Recording:** WAV, MP3, FLAC, OGG, M4A, or a video file whose
     soundtrack is used. With two people, *Take turns* plays the two
     recordings one after the other. *At once* plays them together.
   - **Type it:** Kokoro speaks the text. With two people, mark the
     speakers: `(s1) Did you see it? (s2) I did!`
4. **Length.** *Whole audio* renders the whole recording, up to 40 s.
   *3 s clip* renders one clip, which is quickest for testing a picture.
5. **Prompt.** A sentence about the scene, the people and the camera. It
   steers motion and expression; the lips follow the audio regardless.

The **Settings** popover has the controls that matter on a small card:

| Setting | Default on 8 GB | Notes |
|---|---|---|
| Size | 480 px | 320 px is a fast draft. 640 px is the trained size and wants 12 GB+. |
| Steps | 8 (FusionX) / 40 (base) | Lower is faster. Lip sync survives low step counts; motion detail does not. |
| Prompt strength | 1 (FusionX) / 5 (base) | 1 skips one of three model passes per step. |
| Lip-sync strength | 2 (FusionX) / 4 (base) | Raise it if the mouth lags the audio. |
| TeaCache | off (FusionX) / on (base) | Skips near-duplicate steps. Only worth it at 40 steps. |
| Tiled VAE | on | Turn it off only on 12 GB+. |
| Text encoder on CPU | on | Turn it off only on 16 GB+. |

Every finished clip lands in the feed and the Library with its full recipe.
**Reuse settings** loads the picture, audio and settings back into the bar.
Each running job has a **Log** button that shows the engine's own output.
If a render runs out of GPU memory, the job card says so and suggests what
to turn down.

---

## Tests

```bash
python tests/run.py            # gate, units, api, ui
python tests/run.py gate       # compile, script parse, ids, wiring
```

- **gate** compiles the Python, parses the page's script with node, and
  checks that every element id the script uses exists and every control has
  a listener.
- **units** cover request checking and the engine command line. They also
  check that every flag the studio passes exists in the real
  `generate_multitalk.py` parser, and that the clip arithmetic matches the
  engine's own loop. The preflight verdicts, the weight set, folder
  expansion and path safety are covered too.
- **api** runs the real server against `tests/fake_engine/`, a stand-in that
  takes the real command line and prints the real engine's log lines. It
  covers uploads, renders, the queue, cancel, failures, and resumable
  downloads against a stand-in HuggingFace.
- **ui** drives the page in Chromium through Playwright, and skips itself
  when Playwright is missing.

The engine patches have their own CPU test: `python
../MultiTalk/tests/test_lowvram_patches.py`. It needs torch.

What the tests cannot cover is the real model on a real GPU. The weights
could not be downloaded where this was built. The first real render on the
RTX 4060 is the remaining check.

---

## What changed in the engine

All changes are in `../MultiTalk` and are marked `MultiTalk Studio` in the
code. Upstream behaviour is unchanged unless a new option is used.

- **Tiled VAE** (`wan/modules/vae.py`). Spatial tiles with linear blending
  on encode and decode, via `--vae_tile` and `--vae_tile_overlap`.
- **Smaller sizes** (`wan/configs/__init__.py`,
  `wan/utils/multitalk_utils.py`). `multitalk-360` (480 px) and
  `multitalk-240` (320 px) buckets, scaled from the 640 px table.
- **No flash-attn needed** (`wan/modules/attention.py`). `flash_attention`
  falls back to PyTorch SDPA, with the same lengths and padding. The audio
  cross-attention falls back from xformers the same way.
- **No xfuser needed.** Its import is optional for single-GPU runs.
- **ffmpeg from imageio-ffmpeg** when none is on PATH, and `-y` on the
  audio-crop call so a leftover file cannot hang a render on a prompt.
- **Progress markers.** `[clips] total=…` and `[clip] n` log lines feed
  the progress bar.
- **Short audio is padded** to one clip's length instead of failing an
  assertion. M4A and AAC are decoded through ffmpeg.
- **`--kokoro_dir`** points TTS at the downloaded weights.
- **Runs on Python 3.11 and 3.12.** Upstream imported `inspect.ArgSpec`,
  which Python 3.11 removed, so the engine crashed at start on anything
  newer than 3.10. The import was unused and is gone.
- **Imports without a GPU.** The T5 module asked CUDA for a device at import
  time. It now asks when the encoder is built, so Setup's last step can
  import the whole engine to prove it loads.
