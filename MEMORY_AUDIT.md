# 8 GB memory audit — 9 October 2026

## Fixed

- The tiled decoder accumulated its entire floating-point video on CUDA
  and allocated another full canvas for normalization. With model offloading
  enabled it now blends tiles into host RAM and normalizes in place.
- Full reference-padding buffers survived into sampling. They are now
  released immediately after encoding.
- The CPU wav2vec2 model survived into DiT/T5 loading. Its owner references
  are now released when audio preprocessing finishes.
- Documentation no longer promises a full render on an 8 GB VRAM or
  8 GB system-RAM machine based only on small-model tests.

## Validation

- Studio gate, unit and API suites: **177 checks passed**.
- `OMP_NUM_THREADS=2 python tests/test_lowvram_patches.py`, from `MultiTalk/`:
  **26 checks passed**. Host-accumulated tiles exactly match the previous
  tiled pixel values on CPU with random weights.
- The repository's tiny-engine harness completed the real generation loop
  and ffmpeg mux with random weights: a five-frame 288×352 MP4 with audio,
  one diffusion step, tiled VAE, CPU offload enabled. ffprobe confirmed both
  streams and the expected 0.2-second duration. This is an execution check,
  not a useful avatar or a real-checkpoint GPU benchmark.

No CUDA device or real 14B checkpoint was available. Full-size GPU behavior,
visual quality, long-clip RAM usage and Windows paging still need validation.

## Apply

Update both `MultiTalk/` and `multitalk-studio/`; preserve weights and data.
Restart the studio. Start with INT8 FusionX, 360 render size, persistent DiT
parameters 0, CPU T5 and tiled VAE. Try one short clip with other AI engines
closed. System RAM remains important; 32 GB is a starting point, not a fit
guarantee. Capture the final engine error if the short clip fails.
