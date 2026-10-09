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
- Second pass: VRAM-managed DiT wrappers now offload before VAE decoding;
  persistent weights previously overlapped decoding. Sampling arguments,
  guidance predictions and TeaCache residuals are also released there
  instead of surviving decode and the next clip's reference encode.
- Second pass: preflight messages now describe memory estimates and possible
  failure; they no longer guarantee an 8 GB fit or sufficient system RAM.

## Validation

- Studio gate, unit and API suites: **177 checks passed**.
- `OMP_NUM_THREADS=2 python tests/test_lowvram_patches.py`, from `MultiTalk/`:
  **26 checks passed**. Host-accumulated tiles exactly match the previous
  tiled pixel values on CPU with random weights.
- `OMP_NUM_THREADS=2 python tests/test_pipeline_lifecycle.py`: **6 tests
  passed**, exercising the actual generate loop with instrumented CPU
  components. Covers two- and three-pass guidance, APG on/off, continuation,
  TeaCache cleanup and wrapped-model offloading. Weak references prove that
  sampling tensors are released before decode. Running the original four
  loop tests against the first-pass generate method reproduced four failures.
- The repository's tiny-engine harness completed the real generation loop
  and ffmpeg mux with random weights: a five-frame 288×352 MP4 with audio,
  one diffusion step, tiled VAE, CPU offload enabled. ffprobe confirmed both
  streams and the expected 0.2-second duration. This is an execution check,
  not a useful avatar or a real-checkpoint GPU benchmark.
  The second-pass repeat also enabled `--num_persistent_param_in_dit 0`
  to exercise the real wrapper-management path and completed with the same
  frame count, dimensions, duration and both media streams.

No CUDA device or real 14B checkpoint was available. Full-size GPU behavior,
visual quality, long-clip RAM usage and Windows paging still need validation.

## Apply

Update both `MultiTalk/` and `multitalk-studio/`; preserve weights and data.
Restart the studio. Start with INT8 FusionX, 360 render size, persistent DiT
parameters 0, CPU T5 and tiled VAE. Try one short clip with other AI engines
closed. System RAM remains important; 32 GB is a starting point, not a fit
guarantee. Capture the final engine error if the short clip fails.
