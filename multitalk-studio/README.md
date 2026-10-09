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
GB of VRAM. This app targets an **8 GB GPU** with the following reductions;
these are not proof that a full model render fits every 8 GB installation:

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

**System RAM matters as much as VRAM.** The GPU borrows the INT8 model and
text encoder a layer at a time, so a render wants around 29 GB of RAM to
keep them all cached. The engine maps both from their files read-only, so
memory mapping reduces duplicate allocations, but activations, decoded
frames and other models still consume RAM. **32 GB system RAM is the
practical starting point; 8 GB system RAM is not a supported promise.**
Keep weights on an SSD. Memory pressure can cause heavy paging or failure.

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

Already downloaded the models? Set **Settings → weights folder** to the
folder containing `Wan2.1-I2V-14B-480P`, `MeiGen-MultiTalk`,
`chinese-wav2vec2-base` and, if used, `Kokoro-82M`. Rendering passes those
locations to the engine. Existing finished files are kept. Setup and
**Download set** also reuse matching Hugging Face cache snapshots before
contacting the Hub, including wav2vec's specific `refs/pr/1` revision.
`HF_HUB_CACHE`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`
and the standard `~/.cache/huggingface/hub` layout are supported. Same-drive
reuse uses hard links; copying across drives needs additional disk space.
No cache originals are removed or overwritten.

A tokenizer folder must contain its config and vocabulary. Empty files,
unfinished `.part`/`.incomplete` downloads and Git LFS pointer stubs do not
count as ready. Known byte counts are checked when files are adopted from a
listed download; large cached weights below half their published estimate
are rejected. These checks do not certify model contents: **Test the engine**
is still needed to validate a complete real render.

The last verified paths are saved and reused. If a location stops working,
the app tries recovery once for that changed state and remembers an
unsuccessful drive search across restarts. Normal polling and package
refreshes do not repeat the search. **Engine → Recheck** explicitly allows
another recovery attempt; changing a path also permits a new check.

flash-attn and xfuser are **not** installed. flash-attn has no Windows
wheels and xfuser is only for multi-GPU runs. The engine falls back to
PyTorch's own fused attention without them.

---

## Making a clip

1. **Picture.** A photo or drawing of the people. Cartoons work. It is
   centre-cropped to the video's shape (below), so keep the faces away
   from the edges.
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
| Video | 720 × 360 | The finished file's exact size: **720 × 360** (landscape) or **720 × 1280** (portrait). Nothing else is offered. |
| Render at | 480 px | 320 px is a fast draft. 640 px is the trained size and wants 12 GB+. |
| Steps | 8 (FusionX) / 40 (base) | Lower is faster. Lip sync survives low step counts; motion detail does not. |
| Prompt strength | 1 (FusionX) / 5 (base) | 1 skips one of three model passes per step. |
| Lip-sync strength | 2 (FusionX) / 4 (base) | Raise it if the mouth lags the audio. |
| TeaCache | off (FusionX) / on (base) | Skips near-duplicate steps. Only worth it at 40 steps. |
| Tiled VAE | on | Turn it off only on 12 GB+. |
| Text encoder on CPU | on | Turn it off only on 16 GB+. |

### Why "render at" and "video" are two settings

MultiTalk draws at fixed size buckets whose sides are multiples of 32, so
neither output size can be drawn directly. 360 is not a multiple of 32, and
720 × 1280 would need about 33 GB of VRAM. So the engine renders the bucket
with the output's shape (`--bucket_ratio`), then the studio resizes the
finished clip to the exact pixels with ffmpeg (Lanczos, fill and centre-crop,
H.264 CRF 18, sound copied):

| Video | Render at 320 px | Render at 480 px (8 GB default) | Render at 640 px |
|---|---|---|---|
| 720 × 360 | 448 × 224 → ×1.6 | 672 × 352 → ×1.07 | 896 × 448 → ×0.8 |
| 720 × 1280 | 224 × 416 → ×3.2 | 352 × 640 → ×2 | 448 × 832 → ×1.6 |

On an 8 GB card, 720 × 360 is drawn at nearly its full size. 720 × 1280 is
drawn at about half size and upscaled 2×, with a light sharpen. It is the
right size and shape, but softer than a native render. The job card shows
"Resizing to 720 × 360" as its last step, and the clip's details list both
the final size and the size it was drawn at.

Every finished clip lands in the feed and the Library with its full recipe.
**Reuse settings** loads the picture, audio and settings back into the bar.
Each running job has a **Log** button that shows the engine's own output.
If a render runs out of GPU memory, the job card says so, suggests what
to turn down, and names any other program holding memory on the card.

---

## Does it actually work?

Every row on the Engine page can read ok while the first render still
fails. The weights might never have finished downloading, PyTorch might
not see the card, or the engine might crash on its first step. So the
Engine page has **Test the engine**. It renders one 3-second clip of
MultiTalk's own two-voice example, a man and a woman taking turns, then
checks each step:

| Step | What it checks |
|---|---|
| The engine environment imports the engine | The engine's code loads, and its argument parser runs |
| PyTorch can see an NVIDIA card | The card's name and memory |
| Every weight file is whole | Each file is present and at least half its published size, so a cut-off download fails |
| A 3-second two-voice clip renders | Timed: the first real number for your card |
| The video has frames, not blank, and they move | A black or frozen clip fails |
| Both voices are audible, each in its turn | Voice 1, then voice 2; silence fails |

It stops at the first step that breaks and says what to fix. When it
passes, it also estimates how long 10 seconds of speech takes on your
machine. The test clip goes into the Library so you can watch it.

---

## Tests

```bash
python tests/run.py            # gate, units, reuse, api, ui
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
- **reuse** verifies cache reuse without network requests and checks that
  failed location searches are remembered until a path changes or Recheck
  is explicitly requested.
- **api** runs the real server against `tests/fake_engine/`, a stand-in that
  takes the real command line and prints the real engine's log lines. It
  covers uploads, renders, the queue, cancel, failures, and resumable
  downloads against a stand-in HuggingFace.
- **ui** drives the page in Chromium through Playwright, and skips itself
  when Playwright is missing.

GitHub CI runs these checks on Python 3.10 and 3.13, with the browser tests
on 3.10. It installs CPU PyTorch to execute the real environment probe;
no trained model weights or CUDA packages are downloaded.

The engine patches have their own CPU test: `python
../MultiTalk/tests/test_lowvram_patches.py`. It needs torch. Run
`python ../MultiTalk/tests/test_pipeline_lifecycle.py` as well to check
sampling cleanup and offloading at the decode boundary.

**`tests/tiny_engine/`** runs the real MultiTalk code end to end on a CPU,
with tiny random weights instead of the 29 GB set:

- **Real:** argument parsing, the two-voice audio preparation, wav2vec2,
  the full generation loop with the two-person masks, the DiT class, the
  Wan VAE at full size, and the ffmpeg mux.
- **Stubbed:** only the text and image encoders.

Point the studio's engine folder at it, and its weights folder at one
built by `tests/tiny_engine/make_weights.py`. A render from the page then
gives a real MP4 with the real two-voice soundtrack. The picture is noise,
because the weights are random. This found the text-encoder-on-CPU crash
listed below. At 4 CPU cores, one 81-frame clip takes about 6 minutes,
almost all of it the VAE.

What these tests cannot cover is the real model on a real GPU. The weights
could not be downloaded where this was built. **Test the engine** on the
RTX 4060 is the remaining check.

---

## What changed in the engine

All changes are in `../MultiTalk` and are marked `MultiTalk Studio` in the
code. The memory changes also apply to existing offloading options.

- **Tiled VAE** (`wan/modules/vae.py`). Spatial tiles with linear blending
  on encode and decode, via `--vae_tile` and `--vae_tile_overlap`.
- **Host accumulation for tiled decode.** With offloading enabled, each
  decoded tile is copied to CPU before blending. The full pixel canvas and
  its normalization no longer need CUDA storage. Reference padding buffers
  are released before sampling, and wav2vec2 is released before model loading.
- **Release sampling memory before decode.** Persistent DiT wrappers now
  offload before the VAE runs. Per-clip conditioning, guidance and TeaCache
  tensors are released at the same boundary, including on streaming clips.
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
- **The text encoder works on the CPU** (`wan/multitalk.py`). With `--t5_cpu`,
  the 8 GB default, upstream passed the prompt features as `[[tensor]]`.
  The model died on its first step with "'list' object has no attribute
  'dtype'". The tiny-engine run found it.
- **`--bucket_ratio`** (`wan/multitalk.py`, `generate_multitalk.py`)
  picks the size bucket by the wanted height / width instead of the
  picture's. The picture is then centre-cropped to it, as for any bucket.
  The studio uses this for the 720 × 360 and 720 × 1280 outputs.
- **The INT8 weights load once into RAM** (`wan/utils/lowmem_load.py`,
  used by `wan/multitalk.py` and `wan/modules/t5.py`). Windows reserves
  every allocation against RAM + page file up front, and upstream's loading
  reserved far more than the files: safetensors' `load_file` maps the file
  copy-on-write (the whole map counts) and then copies the tensors out, and
  `optimum.quanto.requantize` builds the model empty in float32 (four times
  the INT8 file) before loading into it. On a 32 GB PC that stopped a render
  at "Loading the model into RAM" with exit code 3221225477 or "The paging
  file is too small (os error 1455)". Now the files are mapped read-only:
  the tensors are the file's own pages, which count against neither RAM nor
  the page file, and Windows re-reads them from disk when RAM is short. That
  reduces host-memory pressure; it does not establish that an 8 GB RAM PC can
  finish a real render.
- **Encode and decode are announced.** `[clip] encoding…` and
  `[clip] decoding…` lines let the job card say "reading the picture" and
  "decoding the frames" instead of sitting on the last step.
