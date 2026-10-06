"""A stand-in for MultiTalk's generate_multitalk.py.

It takes the real command line (the flags the studio builds are checked
against the real parser's names), reads the input JSON the studio wrote,
checks every file it names exists, and prints the lines the real engine
prints — logging, the patched [clips]/[clip] markers, tqdm bars redrawn
with carriage returns — then writes <save_file>.mp4.

Knobs (environment): FAKE_STEP_DELAY seconds per step (default 0.02),
FAKE_FAIL=oom|crash, FAKE_FRAMES audio length in frames (default 140).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
for name in ("--ckpt_dir", "--wav2vec_dir", "--kokoro_dir", "--quant",
             "--quant_dir", "--input_json", "--audio_save_dir", "--save_file",
             "--size", "--mode", "--audio_mode", "--offload_model"):
    p.add_argument(name)
for name in ("--frame_num", "--motion_frame", "--sample_steps",
             "--num_persistent_param_in_dit", "--vae_tile",
             "--vae_tile_overlap", "--base_seed"):
    p.add_argument(name, type=int)
for name in ("--sample_shift", "--sample_text_guide_scale",
             "--sample_audio_guide_scale", "--color_correction_strength",
             "--teacache_thresh"):
    p.add_argument(name, type=float)
p.add_argument("--lora_dir", nargs="+")
for flag in ("--t5_cpu", "--use_teacache", "--use_apg"):
    p.add_argument(flag, action="store_true")
args = p.parse_args()    # an unknown flag exits 2, as the real parser would

out_dir = Path(args.input_json).parent
(out_dir / "argv.json").write_text(json.dumps(sys.argv[1:]))
data = json.loads(Path(args.input_json).read_text())

def say(msg):
    print(msg, flush=True)

seed = args.base_seed if args.base_seed >= 0 else 777
say(f"[2026-01-01 00:00:00] INFO: Generation job args: Namespace(task='multitalk-14B', size='{args.size}', base_seed={seed})")

need = [data["cond_image"]]
for k, v in (data.get("cond_audio") or {}).items():
    if v != "None":
        need.append(v)
tts = data.get("tts_audio")
if tts:
    need += [tts["human1_voice"]] + ([tts["human2_voice"]] if "human2_voice" in tts else [])
    if args.audio_mode != "tts":
        say("ERROR: tts input without --audio_mode tts"); sys.exit(3)
for path in need + [args.ckpt_dir, args.wav2vec_dir, args.quant_dir]:
    if not os.path.exists(path):
        say(f"FileNotFoundError: [Errno 2] No such file or directory: '{path}'")
        sys.exit(1)
if "/" not in data["cond_image"] or "\\" in data["cond_image"]:
    say("ERROR: cond_image must use forward slashes"); sys.exit(3)

say("[2026-01-01 00:00:01] INFO: Creating MultiTalk pipeline.")
say("[2026-01-01 00:00:02] INFO: loading /weights/Wan2.1_VAE.pth")
if os.environ.get("FAKE_FAIL") == "crash":
    say("Traceback (most recent call last):")
    say("ModuleNotFoundError: No module named 'misaki'")
    sys.exit(1)
say("[2026-01-01 00:00:03] INFO: Generating video ...")
frames = int(os.environ.get("FAKE_FRAMES", "140"))
clips = 1 if args.mode == "clip" else 1 + max(1, -(-(frames - 81) // 56))
say(f"[2026-01-01 00:00:03] INFO: [clips] total={clips} frames={frames} steps={args.sample_steps}")
delay = float(os.environ.get("FAKE_STEP_DELAY", "0.02"))
for c in range(clips):
    say(f"[2026-01-01 00:00:04] INFO: [clip] {c + 1} start_frame={c * 56} frames=81")
    for s in range(args.sample_steps + 1):
        sys.stderr.write(f"\r {int(s / args.sample_steps * 100)}%|##| {s}/{args.sample_steps} [00:0{s}<00:00, 1.00s/it]")
        sys.stderr.flush()
        time.sleep(delay)
        if os.environ.get("FAKE_FAIL") == "oom" and c == 0 and s == 2:
            sys.stderr.write("\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
            sys.exit(1)
    sys.stderr.write("\n")
say(f"[2026-01-01 00:00:09] INFO: Saving generated video to {args.save_file}.mp4")
sys.stderr.write("\rSaving video: 100%|##| 81/81\n")
target = Path(args.save_file + ".mp4")
src = Path(__file__).with_name("sample.mp4")
if src.exists():
    shutil.copy(src, target)
else:
    target.write_bytes(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 512)
say("[2026-01-01 00:00:10] INFO: Finished.")
