"""The pure logic: request checking, the engine's command line and input
file, reading its output, clip arithmetic, the preflight verdict, the weight
set and path safety."""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

from harness import ROOT, Suite, fake_weights

import bootstrap                                    # noqa: E402
import engine                                       # noqa: E402
import manager  # noqa: E402
import selftest                                      # noqa: E402

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

    # -- the two output sizes ------------------------------------------------
    s.equal("two outputs only: 720 × 360 and 720 × 1280",
            sorted((o["w"], o["h"]) for o in engine.OUTPUTS.values()),
            [(720, 360), (720, 1280)])
    s.equal("720 × 360 is the default", engine.DEFAULT_OUTPUT, "720x360")
    ja = engine.argv(cfg, engine.normalize(dict(BASE), cfg), Path("/j/in.json"),
                     Path("/j/a"), Path("/c/x"))
    jp = engine.argv(cfg, engine.normalize(dict(BASE, output="720x1280"), cfg),
                     Path("/j/in.json"), Path("/j/a"), Path("/c/x"))
    s.equal("landscape asks the engine for the 2:1 bucket",
            ja[ja.index("--bucket_ratio") + 1], "0.5000")
    s.equal("portrait asks for the 9:16 bucket",
            jp[jp.index("--bucket_ratio") + 1], "1.7778")
    s.fails_with("any other output size is refused",
                 lambda: engine.normalize(dict(BASE, output="1280x720"), cfg),
                 engine.BadRequest, "720")
    ra = engine.resize_args("ffmpeg", Path("a.mp4"), Path("b.mp4"), 720, 360, 1.07)
    vf = ra[ra.index("-vf") + 1]
    s.check("the resize fills the frame, crops to the exact size, keeps the sound",
            "scale=720:360:force_original_aspect_ratio=increase" in vf
            and "crop=720:360" in vf and ra[ra.index("-c:a") + 1] == "copy", vf)
    s.check("a near-1x resize is not sharpened", "unsharp" not in vf)
    rp = engine.resize_args("ffmpeg", Path("a.mp4"), Path("b.mp4"), 720, 1280, 2.0)
    s.check("a 2x upscale gets a light sharpen",
            "unsharp" in rp[rp.index("-vf") + 1])

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
    ph = engine.new_state()
    for ln in ("INFO: [clips] total=2 frames=140 steps=8",
               "INFO: [clip] 1 start_frame=0 frames=81",
               "INFO: [clip] encoding the reference frames"):
        engine.read_line(ph, ln)
    s.equal("the picture being read is named on the card", ph["stage"],
            "Clip 1 of 2 · reading the picture")
    for ln in [f" 50%|##| {i}/8 [00:01<00:01]" for i in range(9)] + \
            ["INFO: [clip] decoding the frames"]:
        engine.read_line(ph, ln)
    s.equal("and so is decoding, after the last step", ph["stage"],
            "Clip 1 of 2 · decoding the frames")
    oom = engine.new_state()
    engine.read_line(oom, "torch.OutOfMemoryError: CUDA out of memory.")
    s.check("out of memory is explained in plain words",
            "smaller size" in engine.failure(oom, 1))
    miss = engine.new_state()
    engine.read_line(miss, "ModuleNotFoundError: No module named 'misaki'")
    s.check("a missing package points at the Engine page",
            "Engine page" in engine.failure(miss, 1))
    cpu = engine.new_state()
    engine.read_line(cpu, "AssertionError: Torch not compiled with CUDA enabled")
    s.check("a CPU build of PyTorch is named, with Reinstall",
            "CPU build" in engine.failure(cpu, 1)
            and "Reinstall" in engine.failure(cpu, 1))
    drv = engine.new_state()
    engine.read_line(drv, "RuntimeError: Found no NVIDIA driver on your system.")
    s.check("a missing driver is named as the driver",
            "driver" in engine.failure(drv, 1))

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

    # -- setup's last step imports the engine, not just its packages ---------
    import shutil as _sh
    fake = ROOT / "tests" / "fake_engine"
    ok, why = bootstrap.check_engine_loads(sys.executable, fake)
    s.check("the load check passes an engine that imports and parses", ok, why)
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "eng"
        _sh.copytree(fake, broken)
        src_ = (broken / "generate_multitalk.py").read_text()
        (broken / "generate_multitalk.py").write_text(
            "from inspect import ArgSpec\n" + src_)
        ok, why = bootstrap.check_engine_loads(sys.executable, broken)
        s.check("and fails one that cannot import, naming why",
                not ok and "ArgSpec" in why, why)
    s.check("the real engine no longer imports inspect.ArgSpec (gone in 3.11)",
            "from inspect import ArgSpec" not in
            (REAL_ENGINE / "wan/multitalk.py").read_text())
    s.check("importing the real T5 module no longer needs a GPU",
            "device=torch.cuda.current_device()," not in
            (REAL_ENGINE / "wan/modules/t5.py").read_text())

    # -- progress reporting ----------------------------------------------------
    import subprocess as _sp
    ver = _sp.run([sys.executable, "-m", "pip", "--version"], capture_output=True,
                  text=True).stdout.split()[1]
    want = tuple(int(x) for x in ver.split(".")[:2]) >= (24, 1)
    bootstrap._PIP_RAW_OK.clear()
    s.equal(f"raw pip progress is detected by asking pip itself (pip {ver})",
            bootstrap.pip_has_raw_progress(sys.executable), want)
    real_stream = bootstrap.stream
    try:
        def fake_stream(cmd, on_line, **kw):
            for ln in ("Downloading torch-2.4.1-cp311-none.whl (797 MB)",
                       "Progress 0 of 800", "Progress 200 of 800",
                       "Progress 400 of 800", "Progress 800 of 800",
                       "Installing collected packages: torch"):
                on_line(ln)
            return 0
        bootstrap.stream = fake_stream
        bootstrap._PIP_RAW_OK["fake-python"] = True
        calls = []
        bootstrap.pip_install("fake-python", ["torch"], lambda m: None,
                              lambda pct, d: calls.append(pct))
        s.equal("every pip progress line reaches the bar, none dropped",
                calls, [0.0, 25.0, 50.0, 100.0, None])
    finally:
        bootstrap.stream = real_stream
        bootstrap._PIP_RAW_OK.pop("fake-python", None)
    tasks = manager.Tasks()
    big = tasks.add(manager.Task("download", "the 16 GB model"))
    for i in range(30):
        tasks.add(manager.Task("download", f"small {i}")).set(state="done")
    for i in range(3):
        tasks.add(manager.Task("download", f"running {i}"))
    shown = tasks.visible()
    s.check("the task list never drops a running download, however old",
            big in shown and sum(t.state == "running" for t in shown) == 4)
    s.equal("and still caps the finished ones",
            sum(t.state == "done" for t in shown), 25)
    t = manager.Task("download", "x")
    s.check("a task reports bytes and whether it is merely busy",
            {"got", "total", "busy"} <= set(t.view()))

    real_run = bootstrap._run
    try:
        class Out:
            stdout = ("1234, C:\\Program Files\\Steam, Inc\\steam.exe, 812\n"
                      "99, chrome.exe, 1450\n")
        bootstrap._run = lambda *a, **k: Out()
        real_which = bootstrap.shutil.which
        bootstrap.shutil.which = lambda n: "/usr/bin/nvidia-smi"
        procs = bootstrap.gpu_processes()
    finally:
        bootstrap._run = real_run
        bootstrap.shutil.which = real_which
    s.equal("other programs on the card are read, commas in paths and all",
            [(p["name"], p["mb"]) for p in procs],
            [("steam.exe", 812), ("chrome.exe", 1450)])

    # -- the self-test's judgements ------------------------------------------
    import random as _r
    flat = [bytes([128]) * 64 for _ in range(81)]
    s.check("blank frames fail the self-test",
            not selftest.judge_frames(selftest.frame_stats(flat), 81)[0])
    def noise(seed):
        g = _r.Random(seed)
        return bytes(g.randrange(256) for _ in range(64))
    still = [noise(1)] * 81
    s.check("a still picture fails it too",
            not selftest.judge_frames(selftest.frame_stats(still), 81)[0])
    moving = [noise(i) for i in range(81)]
    s.check("frames with contrast that change pass",
            selftest.judge_frames(selftest.frame_stats(moving), 81)[0])
    s.check("too few frames fail",
            not selftest.judge_frames(selftest.frame_stats(moving[:20]), 81)[0])
    rate = 8000
    tone = [int(8000 * math.sin(i / 3)) for i in range(rate)]
    hush = [0] * rate
    s.check("two voices, each in its turn, pass",
            selftest.judge_voices(tone + tone + hush, rate, [1, 1], 3)[0])
    ok, why = selftest.judge_voices(tone + hush + hush, rate, [1, 1], 3)
    s.check("the second voice silent in its turn fails, naming it",
            not ok and "Voice 2" in why, why)
    s.check("a silent soundtrack fails",
            not selftest.judge_voices(hush * 3, rate, [1, 1], 3)[0])
    s.check("no soundtrack fails",
            not selftest.judge_voices([], rate, [1, 1], 3)[0])
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp)
        for item in bootstrap.model_set(cfg):
            t = bootstrap.model_path(w, item)
            if item["prefix"]:
                t.mkdir(parents=True, exist_ok=True)
                (t / "x.json").write_text("{}")
            else:
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes(b"placeholder")
        ok, why = selftest.weights_whole(w, dict(cfg))
        s.check("placeholder weights are caught by size, not passed as present",
                not ok and "Too small" in why, why[:120])
    ex = REAL_ENGINE.joinpath(*selftest.EXAMPLE)
    s.check("MultiTalk's two-voice example the self-test uses is in the repo",
            all((ex / f).is_file() for f in selftest.EXAMPLE_FILES.values()))

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

    # -- a moved repo folder: stale saved paths are found again --------------
    real_repo, real_app = bootstrap.REPO_DIR, bootstrap.APP_DIR
    real_default = bootstrap.DEFAULT_CONFIG["engine_dir"]
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "Multi-Talk-main"          # renamed on extraction
        app = repo / real_app.name
        eng = repo / "MultiTalk"
        (eng / "weights").mkdir(parents=True)
        (eng / "generate_multitalk.py").write_text("")
        vpy = app / "engine-venv" / "Scripts" / "python.exe"
        vpy.parent.mkdir(parents=True)
        vpy.write_text("")
        bootstrap.REPO_DIR, bootstrap.APP_DIR = repo, app
        bootstrap.DEFAULT_CONFIG["engine_dir"] = str(eng)
        try:
            old = "C:\\AI\\Multi-Talk\\MultiTalk"
            moved = dict(bootstrap.DEFAULT_CONFIG, engine_dir=old,
                         weights_dir=old + "\\weights",
                         python="C:\\AI\\Multi-Talk\\" + real_app.name
                                + "\\engine-venv\\Scripts\\python.exe")
            notes = bootstrap.heal_paths(moved)
            s.equal("a stale engine path is rebased onto the renamed repo",
                    moved["engine_dir"], str(eng))
            s.equal("a stale weights path follows it",
                    moved["weights_dir"], str(eng / "weights"))
            s.equal("a moved engine Python is rebased with the venv",
                    moved["python"], str(vpy))
            s.check("each repair is reported", len(notes) == 3, str(notes))
            s.check("a healed config is left alone",
                    bootstrap.heal_paths(moved) == [])
            gone = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(eng),
                        python="E:\\elsewhere\\python312\\python.exe")
            bootstrap.heal_paths(gone)
            s.equal("a vanished Python path is cleared, not kept",
                    gone["python"], "")
            stale_default = dict(bootstrap.DEFAULT_CONFIG,
                                 engine_dir="/old/place/whatever/MultiTalk")
            bootstrap.heal_paths(stale_default)
            s.equal("a saved default engine path comes back to the repo's",
                    stale_default["engine_dir"], str(eng))
        finally:
            bootstrap.REPO_DIR, bootstrap.APP_DIR = real_repo, real_app
            bootstrap.DEFAULT_CONFIG["engine_dir"] = real_default

    # -- the start-up search: finds the engine anywhere, prefers the weights -
    s.check("the real engine checkout is recognised",
            bootstrap.is_engine_dir(REAL_ENGINE))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def make(rel: str) -> Path:
            c = root / rel
            (c / "wan").mkdir(parents=True)
            (c / "weights").mkdir()
            (c / "generate_multitalk.py").write_text("")
            (c / "wan" / "multitalk.py").write_text("")
            return c
        bare = make("a/MultiTalk")
        rich = make("x/y/z/w/MultiTalk")
        make("a/MultiTalk/src/inner/MultiTalk")        # never walked into
        (root / "Windows" / "MultiTalk" / "wan").mkdir(parents=True)
        (root / "Windows" / "MultiTalk" / "generate_multitalk.py").write_text("")
        (root / "Windows" / "MultiTalk" / "wan" / "multitalk.py").write_text("")
        (root / "b" / "half").mkdir(parents=True)       # entry script only
        (root / "b" / "half" / "generate_multitalk.py").write_text("")
        found = bootstrap.find_engine_installs([root], max_depth=6, budget=10)
        s.equal("the search finds every engine, shallowest first",
                found, [bare, rich])
        s.equal("the search stops at the depth limit",
                bootstrap.find_engine_installs([root], max_depth=3, budget=10),
                [bare])
        wcfg = dict(bootstrap.DEFAULT_CONFIG)
        fake_weights(rich / "weights")
        s.equal("the engine holding the weights is the one chosen",
                bootstrap.pick_engine(found, wcfg), rich)
        real_find = bootstrap.find_engine_installs
        bootstrap.find_engine_installs = lambda: [bare, rich]
        # the real repo's engine must not answer for the lost one: no
        # default folder, no repo to rebase onto
        bootstrap.DEFAULT_CONFIG["engine_dir"] = str(root / "no-default")
        bootstrap.REPO_DIR = root / "no-repo"
        bootstrap.APP_DIR = root / "no-repo" / real_app.name
        try:
            lost = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(root / "gone"),
                        weights_dir=str(root / "gone" / "weights"))
            notes = bootstrap.verify_locations(lost)
            s.equal("a lost engine is found by the search",
                    lost["engine_dir"], str(rich))
            s.equal("and its weights folder with it",
                    bootstrap.weights_dir(lost), rich / "weights")
            s.check("the report says both check out",
                    all("not found" not in ln and "missing" not in ln
                        for ln in bootstrap.location_report(lost)[:2]),
                    str(bootstrap.location_report(lost)))
            quiet = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(root / "gone"))
            bootstrap.verify_locations(quiet, search=False)
            s.equal("no search when asked not to",
                    quiet["engine_dir"], str(root / "gone"))
        finally:
            bootstrap.find_engine_installs = real_find
            bootstrap.DEFAULT_CONFIG["engine_dir"] = real_default
            bootstrap.REPO_DIR, bootstrap.APP_DIR = real_repo, real_app
    row = next(i for i in manager.dependencies(
        dict(bootstrap.DEFAULT_CONFIG, engine_dir="/nowhere/MultiTalk"),
        searching=True) if i["id"] == "engine")
    s.check("while searching, the engine row says so and offers nothing",
            row["state"] == "warn" and "Searching" in row["detail"]
            and row["action"] is None)
    return s
