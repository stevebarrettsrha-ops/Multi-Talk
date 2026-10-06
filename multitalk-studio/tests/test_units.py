"""The pure logic: request checking, the engine's command line and input
file, reading its output, clip arithmetic, the preflight verdict, the weight
set and path safety."""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

from harness import ROOT, Suite

import bootstrap                                    # noqa: E402
import engine                                       # noqa: E402
import manager                                      # noqa: E402

REAL_ENGINE = ROOT.parent / "MultiTalk"
GIB = 1024 ** 3


def engine_clips(frames: int, mode: str) -> int:
    """MultiTalkPipeline.generate()'s loop, step for step."""
    frame_num, motion = 81, 25
    max_frames = frame_num if mode == "clip" else 1000
    n = max(frames, frame_num + 1)
    start, end, clips, last = 0, frame_num, 0, False
    while True:
        clips += 1
        if last:
            break
        start += frame_num - motion
        end = start + frame_num
        if end >= min(max_frames, n):
            last = True
        if max_frames <= frame_num:
            break
    return clips


BASE = {"image": "aaaaaaaaaaaa.png", "audio1": "bbbbbbbbbbbb.wav"}


def run(slow: bool = False) -> Suite:
    s = Suite("units")
    cfg = dict(bootstrap.DEFAULT_CONFIG, weights_dir="/w")

    # -- clip arithmetic matches the engine's own loop -----------------------
    s.check("clips_for() matches the pipeline loop for 0-60 s of audio",
            all(engine.clips_for(f, "streaming") == engine_clips(f, "streaming")
                for f in range(0, 1500, 7)))
    s.equal("clip mode is always one clip", engine.clips_for(900, "clip"), 1)
    s.equal("10 s of speech is five clips (81 frames, then 56 more each)",
            engine.clips_for(250), 5)
    s.check("the patched pipeline logs the same count it computes",
            "[clips] total=" in (REAL_ENGINE / "wan/multitalk.py").read_text())

    # -- every flag the studio passes exists in the real parser --------------
    real = (REAL_ENGINE / "generate_multitalk.py").read_text()
    known = set(re.findall(r'add_argument\(\s*"(--[a-z0-9_]+)"', real))
    job = engine.normalize(dict(BASE, teacache=True, apg=True), cfg)
    flags = {a for a in engine.argv(cfg, job, Path("/j/in.json"), Path("/j/a"),
                                    Path("/c/x")) if a.startswith("--")}
    s.check("every flag the studio builds is one generate_multitalk.py knows",
            flags <= known, f"unknown: {sorted(flags - known)}")

    # -- normalize -----------------------------------------------------------
    s.fails_with("no picture is a clear refusal",
                 lambda: engine.normalize({"audio1": "x.wav"}, cfg),
                 engine.BadRequest, "picture")
    s.fails_with("one person needs a recording",
                 lambda: engine.normalize({"image": "a.png"}, cfg),
                 engine.BadRequest, "recording")
    s.fails_with("three people is refused",
                 lambda: engine.normalize(dict(BASE, people=3), cfg),
                 engine.BadRequest, "one or two")
    s.fails_with("two people typed without (s1)/(s2) is refused with an example",
                 lambda: engine.normalize(dict(BASE, people=2, source="tts",
                                               tts_text="hello", voice1="a",
                                               voice2="b"), cfg),
                 engine.BadRequest, "(s1)")
    s.fails_with("steps out of range say the range",
                 lambda: engine.normalize(dict(BASE, steps=500), cfg),
                 engine.BadRequest, "between 2 and 60")
    s.fails_with("a word for a number is refused",
                 lambda: engine.normalize(dict(BASE, shift="lots"), cfg),
                 engine.BadRequest, "number")
    s.fails_with("NaN is refused", lambda: engine.normalize(
        dict(BASE, text_scale=float("nan")), cfg), engine.BadRequest, "")
    s.fails_with("a tiny VAE tile is refused",
                 lambda: engine.normalize(dict(BASE, vae_tile=4), cfg),
                 engine.BadRequest, "seams")
    s.fails_with("an unknown size is refused",
                 lambda: engine.normalize(dict(BASE, size="multitalk-9000"), cfg),
                 engine.BadRequest, "size")
    j = engine.normalize(dict(BASE), cfg)
    s.check("FusionX defaults fill in: 8 steps, guidance 1 / 2, shift 2",
            (j["steps"], j["text_scale"], j["audio_scale"], j["shift"])
            == (8, 1.0, 2.0, 2.0))
    s.check("8 GB defaults: 480 px, tiled VAE, text encoder on the CPU",
            j["size"] == "multitalk-360" and j["vae_tile"] == 32 and j["t5_cpu"])
    jb = engine.normalize(dict(BASE), dict(cfg, precision="int8"))
    s.check("the base model's defaults are its own: 40 steps, 5 / 4, TeaCache",
            (jb["steps"], jb["text_scale"], jb["audio_scale"], jb["teacache"])
            == (40, 5.0, 4.0, True))
    s.equal("an empty prompt gets a neutral one", j["prompt"],
            "A person is talking.")
    s.equal("seed left empty means random", j["seed"], -1)

    # -- the input JSON ------------------------------------------------------
    up, voices, adir = Path("/u"), Path("/v"), Path("/a")
    d = engine.input_json(j, up, voices, adir)
    s.check("one person: cond_image and person1, forward slashes",
            d["cond_image"].endswith("/u/aaaaaaaaaaaa.png")
            and list(d["cond_audio"]) == ["person1"]
            and "\\" not in d["cond_image"])
    two = engine.normalize(dict(BASE, people=2, audio2="", audio_type="para"), cfg)
    d2 = engine.input_json(two, up, voices, adir)
    s.check("two people with one recording: the other is the string None",
            d2["cond_audio"]["person2"] == "None" and d2["audio_type"] == "para")
    tts = engine.normalize(dict(image="i.png", source="tts", people=2,
                                tts_text="(s1) hi (s2) yo", voice1="af_heart",
                                voice2="am_adam"), cfg)
    d3 = engine.input_json(tts, up, voices, adir)
    s.check("typed speech: tts_audio with both voices and an empty cond_audio",
            d3["tts_audio"]["human2_voice"].endswith("/v/am_adam.pt")
            and d3["cond_audio"] == {})

    # -- the command line ----------------------------------------------------
    a = engine.argv(cfg, j, Path("/j/in.json"), Path("/j/a"), Path("/c/x"))
    val = lambda k: a[a.index(k) + 1]                       # noqa: E731
    s.check("FusionX goes in as the quantised 'LoRA', beside its map",
            val("--lora_dir").endswith(
                "MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors"))
    s.check("INT8 with the DiT fully offloaded",
            val("--quant") == "int8" and val("--num_persistent_param_in_dit") == "0")
    s.check("the 8 GB switches are on the command line",
            "--t5_cpu" in a and val("--vae_tile") == "32")
    ab = engine.argv(dict(cfg, precision="int8"), jb, Path("/j"), Path("/a"),
                     Path("/c"))
    s.check("the base model has no --lora_dir and does use TeaCache",
            "--lora_dir" not in ab and "--use_teacache" in ab)
    s.equal("a random seed is -1", val("--base_seed"), "-1")

    # -- reading the output --------------------------------------------------
    st = engine.new_state()
    lines = ["INFO: Generation job args: Namespace(base_seed=4242, size='x')",
             "INFO: Creating MultiTalk pipeline.",
             "INFO: Generating video ...",
             "INFO: [clips] total=2 frames=140 steps=8",
             "INFO: [clip] 1 start_frame=0 frames=81"]
    lines += [f" 50%|##| {i}/8 [00:01<00:01]" for i in range(9)]
    lines += ["INFO: [clip] 2 start_frame=56 frames=81"]
    lines += [f" 50%|##| {i}/8 [00:01<00:01]" for i in range(9)]
    lines += ["Saving video: 30%|##| 30/140 [00:01<00:01]",
              "INFO: Finished."]
    seen = []
    for ln in lines:
        engine.read_line(st, ln)
        seen.append(st["pct"])
    s.check("progress only ever moves forward", seen == sorted(seen), str(seen))
    s.equal("the seed the engine chose is read back", st["seed"], 4242)
    mid = engine.new_state()
    for ln in lines[:5] + lines[5:14] + lines[14:15] + lines[15:19]:
        engine.read_line(mid, ln)
    s.check("halfway through the second clip reads ~75 % and says so",
            70 <= mid["pct"] <= 80 and mid["stage"].startswith("Clip 2 of 2"),
            f"{mid['pct']} {mid['stage']}")
    s.check("the save bar does not count as sampling",
            st["stage"] == "Finished" and st["pct"] == 99)
    oom = engine.new_state()
    engine.read_line(oom, "torch.OutOfMemoryError: CUDA out of memory.")
    s.check("out of memory is explained in plain words",
            "smaller size" in engine.failure(oom, 1))
    miss = engine.new_state()
    engine.read_line(miss, "ModuleNotFoundError: No module named 'misaki'")
    s.check("a missing package points at the Engine page",
            "Engine page" in engine.failure(miss, 1))

    # -- preflight -----------------------------------------------------------
    peak = bootstrap.peak_ram(cfg)
    v, _ = bootstrap.assess(8 * GIB, 32 * GIB, 500e9, 30e9, peak)
    s.equal("an RTX 4060 with 32 GB RAM is tight, not hard", v, "tight")
    v, _ = bootstrap.assess(8 * GIB, 16 * GIB, 500e9, 30e9, peak)
    s.equal("16 GB of RAM cannot hold the INT8 weights: hard", v, "hard")
    v, _ = bootstrap.assess(4 * GIB, 64 * GIB, 500e9, 30e9, peak)
    s.equal("4 GB of VRAM is hard", v, "hard")
    v, _ = bootstrap.assess(24 * GIB, 64 * GIB, 500e9, 30e9, peak)
    s.equal("24 GB / 64 GB is ok", v, "ok")
    v, notes = bootstrap.assess(8 * GIB, 32 * GIB, 10e9, 30e9, peak)
    s.check("a full disk is hard and says how much is needed",
            v == "hard" and any("needs" in n for n in notes))
    s.check("the RAM peak is honest: over 24 GiB with the INT8 set",
            peak > 24 * GIB, f"{peak / GIB:.1f} GiB")
    s.equal("8 GB gets 480 px, tiling and the CPU text encoder",
            bootstrap.recommended(8 * GIB),
            {"size": "multitalk-360", "t5_cpu": True, "vae_tile": 32})

    # -- the engine check runs in a real interpreter --------------------------
    import sys
    rep = bootstrap.check_engine(sys.executable)
    s.check("the engine check runs and reports torch and what is missing",
            "error" not in rep and isinstance(rep.get("missing"), list)
            and "torch" in rep, str(rep)[:200])

    # -- the weight set ------------------------------------------------------
    items = bootstrap.model_set(cfg)
    names = {i["path"] for i in items}
    s.check("the FusionX set has the files the engine opens by name",
            {"quant_models/quant_model_int8_FusionX.safetensors",
             "quant_models/quantization_map_int8_FusionX.json",
             "quant_models/t5_int8.safetensors", "quant_models/t5_map_int8.json",
             "Wan2.1_VAE.pth", "model.safetensors"} <= names)
    src = (REAL_ENGINE / "wan/multitalk.py").read_text() + \
        (REAL_ENGINE / "wan/modules/t5.py").read_text()
    s.check("those names are the ones the engine builds",
            all(x in src for x in ("quantization_map_{quant}_FusionX.json",
                                   "dit_model_map_{quant}.json",
                                   "t5_{quant}.safetensors")))
    s.check("no 50 GB bf16 shards in the set",
            not any("diffusion_pytorch_model" in n for n in names))
    s.check("wav2vec's safetensors come from the PR branch, as upstream",
            any(i["revision"] == "refs/pr/1" for i in items))
    no_tts = bootstrap.model_set(dict(cfg, want_tts=False))
    s.check("without typed speech there is no Kokoro",
            not any(i["root"] == "Kokoro-82M" for i in no_tts))
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp)
        folder = next(i for i in items if i["prefix"])
        target = bootstrap.model_path(w, folder)
        target.mkdir(parents=True)
        (target / "a.json.part").write_text("x")
        s.check("a folder still downloading is not present",
                not bootstrap.present(w, folder))
        (target / "a.json.part").rename(target / "a.json")
        s.check("a finished folder is present", bootstrap.present(w, folder))

        # path safety
        wcfg = dict(cfg, weights_dir=str(w))
        (w / "keep.pth").write_text("x")
        s.fails_with("delete refuses ../", lambda: manager.delete_model(
            wcfg, "../etc/passwd"), RuntimeError, "not allowed")
        s.fails_with("delete refuses an absolute path", lambda: manager.delete_model(
            wcfg, "/etc/passwd"), RuntimeError, "not allowed")
        s.fails_with("delete refuses backslashes", lambda: manager.delete_model(
            wcfg, "..\\x"), RuntimeError, "not allowed")
        manager.delete_model(wcfg, "keep.pth")
        s.check("delete removes a file under the weights folder",
                not (w / "keep.pth").exists())

    # -- folder items expand from the repo listing ---------------------------
    real_tree = bootstrap.hf_tree
    try:
        bootstrap.hf_tree = lambda c, repo, rev="main": [
            {"path": "google/umt5-xxl/tokenizer.json", "size": 10},
            {"path": "google/umt5-xxl/spiece.model", "size": 20},
            {"path": "Wan2.1_VAE.pth", "size": 99}]
        out = bootstrap.expand(cfg, [i for i in items if i["root"] ==
                                     bootstrap.WAN_DIR])
        s.check("a folder item becomes each of its files",
                {o["path"] for o in out} >= {"google/umt5-xxl/tokenizer.json",
                                             "google/umt5-xxl/spiece.model"})
        vae = next(o for o in out if o["path"] == "Wan2.1_VAE.pth")
        s.equal("sizes come from the repo when it answers", vae["size"], 99)
    finally:
        bootstrap.hf_tree = real_tree
    return s
