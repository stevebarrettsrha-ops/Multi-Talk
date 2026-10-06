"""The HTTP surface, end to end: a real server.py driving the fake engine
through the same subprocess path the real one takes, plus the download
paths against a stand-in HuggingFace."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from harness import (SAMPLE_PNG, SAMPLE_WAV, Suite, Workspace, fake_weights,
                     finish, hub, studio, upload, wait_for)


def gen(url: str, **body) -> requests.Response:
    return requests.post(url + "/api/generate", json=body, timeout=30)


def run(slow: bool = False) -> Suite:
    s = Suite("api")
    with Workspace() as ws:
        weights = ws / "weights"
        fake_weights(weights)
        with studio(ws / "data", weights) as app:
            u = app.url
            st = requests.get(u + "/api/status", timeout=10).json()
            s.check("status says ready with the set on disk",
                    st["ready"] and not st["missing_models"], st.get("blocker"))
            s.check("typed speech is available once Kokoro's voices are there",
                    st["tts_ready"])
            s.equal("the voices are listed",
                    requests.get(u + "/api/voices", timeout=10).json(),
                    ["af_heart", "am_adam"])
            page = requests.get(u + "/", timeout=10)
            s.check("the page is served", page.ok and "MultiTalk Studio" in page.text)

            # -- guard rails -------------------------------------------------
            r = requests.get(u + "/api/status", headers={"Host": "evil.example"},
                             timeout=10)
            s.equal("another host name is refused (DNS rebinding)",
                    r.status_code, 403)
            r = requests.post(u + "/api/config", json={},
                              headers={"Origin": "http://evil.example"}, timeout=10)
            s.equal("a cross-site POST is refused", r.status_code, 403)
            r = requests.post(u + "/api/upload",
                              files={"file": ("x.exe", b"MZ")}, timeout=10)
            s.check("an upload that is neither picture nor audio is refused",
                    r.status_code == 400)
            r = gen(u, image="ffffffffffff.png", audio1="eeeeeeeeeeee.wav")
            s.check("names that were never uploaded are refused",
                    r.status_code == 400 and "no longer on disk" in r.json()["error"])
            r = gen(u, image="../../etc/passwd", audio1="x.wav")
            s.equal("a path for an upload name is refused", r.status_code, 400)
            r = requests.get(u + "/api/upload/..%2Fconfig.json", timeout=10)
            s.equal("uploads cannot be read outside their folder",
                    r.status_code, 404)

            img = upload(u, SAMPLE_PNG)
            wav = upload(u, SAMPLE_WAV)
            s.check("uploads come back typed and renamed",
                    img["kind"] == "image" and wav["kind"] == "audio"
                    and img["name"] != SAMPLE_PNG.name)
            s.check("an upload can be fetched back for its thumbnail",
                    requests.get(u + "/api/upload/" + img["name"],
                                 timeout=10).content == SAMPLE_PNG.read_bytes())
            r = gen(u, image=wav["name"], audio1=wav["name"])
            s.check("audio as the picture is refused", r.status_code == 400)

            # -- one person, end to end -------------------------------------
            r = gen(u, image=img["name"], audio1=wav["name"],
                    prompt="A woman sings in a studio.", seed=1234)
            s.check("generate queues a job", r.ok and len(r.json()["jobs"]) == 1,
                    r.text[:200])
            jid = r.json()["jobs"][0]
            saw = []
            wait_for(lambda: saw.append(
                next(j for j in requests.get(u + "/api/jobs", timeout=10).json()
                     if j["id"] == jid)) or saw[-1]["status"] not in
                ("queued", "running"), 60, 0.1)
            job = saw[-1]
            s.equal("the job finishes", job["status"], "done")
            stages = {x["stage"] for x in saw}
            s.check("the stages a person sees come from the engine's own lines",
                    any(st_.startswith("Clip 1 of 3") for st_ in stages)
                    and any(st_.startswith("Clip 3 of 3") for st_ in stages),
                    str(sorted(stages))[:300])
            pcts = [x["pct"] for x in saw]
            s.check("the bar never walks backwards", pcts == sorted(pcts))
            jdir = ws / "data" / "jobs" / jid
            argv = json.loads((jdir / "argv.json").read_text())
            meta = json.loads((jdir / "input.json").read_text())
            s.check("the engine got the 8 GB command line",
                    "--t5_cpu" in argv and argv[argv.index("--vae_tile") + 1] == "32"
                    and argv[argv.index("--num_persistent_param_in_dit") + 1] == "0")
            s.check("the input file names the uploads by full path",
                    meta["cond_image"].endswith(img["name"])
                    and meta["cond_audio"]["person1"].endswith(wav["name"]))
            s.check("the engine's log is kept with the job",
                    "[clips] total=3" in (jdir / "engine.log").read_text())
            clips = requests.get(u + "/api/clips", timeout=10).json()
            s.equal("one clip in the gallery", len(clips), 1)
            c = clips[0]
            s.check("the gallery keeps the recipe and the seed",
                    c["seed"] == 1234 and c["steps"] == 8 and c["people"] == 1
                    and c["size"] == "multitalk-360" and c["seconds"] == 5.6)
            body = requests.get(u + "/api/clip/" + c["id"], timeout=10)
            s.check("the clip streams back as mp4",
                    body.ok and body.content[4:8] == b"ftyp"
                    and "video/mp4" in body.headers.get("Content-Type", ""))
            s.check("the scratch audio folder is cleaned up",
                    not (jdir / "audio").exists())

            # -- two people, typed --------------------------------------------
            r = gen(u, image=img["name"], people=2, source="tts",
                    tts_text="(s1) Hello. (s2) Hi there!", voice1="af_heart",
                    voice2="am_adam", mode="clip")
            s.check("two people with typed speech queue", r.ok, r.text[:200])
            finish(u)
            jid2 = r.json()["jobs"][0]
            meta = json.loads((ws / "data" / "jobs" / jid2 / "input.json").read_text())
            argv = json.loads((ws / "data" / "jobs" / jid2 / "argv.json").read_text())
            s.check("typed speech reaches the engine as tts with both voices",
                    argv[argv.index("--audio_mode") + 1] == "tts"
                    and meta["tts_audio"]["human2_voice"].endswith("am_adam.pt"))
            r = gen(u, image=img["name"], source="tts", tts_text="hi",
                    voice1="zz_nobody")
            s.check("a voice that is not installed is refused",
                    r.status_code == 400 and "zz_nobody" in r.json()["error"])

            # -- queue, cancel ------------------------------------------------
        with studio(ws / "data2", weights,
                    env={"FAKE_STEP_DELAY": "0.15"}) as app:
            u = app.url
            img, wav = upload(u, SAMPLE_PNG), upload(u, SAMPLE_WAV)
            a = gen(u, image=img["name"], audio1=wav["name"]).json()["jobs"][0]
            b = gen(u, image=img["name"], audio1=wav["name"]).json()["jobs"][0]
            c = gen(u, image=img["name"], audio1=wav["name"]).json()["jobs"][0]
            time.sleep(0.8)
            jobs = {j["id"]: j for j in requests.get(u + "/api/jobs", timeout=10).json()}
            s.check("one render at a time: the others wait their turn",
                    jobs[a]["status"] == "running" and jobs[b]["status"] == "queued"
                    and jobs[b]["position"] == 1 and jobs[c]["position"] == 2,
                    str({k: (v["status"], v.get("position")) for k, v in jobs.items()}))
            r = requests.post(f"{u}/api/jobs/{c}/cancel", timeout=10)
            s.check("a queued job can be removed", r.ok)
            r = requests.post(f"{u}/api/jobs/{a}/cancel", timeout=10)
            s.check("a running job can be stopped", r.ok)
            ok = wait_for(lambda: next(j for j in requests.get(
                u + "/api/jobs", timeout=10).json() if j["id"] == a)["status"]
                == "cancelled", 20)
            s.check("the stopped job says so, and the engine is gone", ok)
            jobs = {j["id"]: j for j in finish(u)}
            s.check("the next one in line runs after it",
                    jobs[b]["status"] == "done" and jobs[c]["status"] == "cancelled")
            s.check("a stopped render leaves no half-written clip",
                    not (ws / "data2" / "clips" / f"{a}.mp4").exists())
            s.equal("cancelling an unknown job is a 404",
                    requests.post(u + "/api/jobs/nope/cancel", timeout=10)
                    .status_code, 404)

        # -- failures read like advice ----------------------------------------
        with studio(ws / "data3", weights, env={"FAKE_FAIL": "oom"}) as app:
            u = app.url
            img, wav = upload(u, SAMPLE_PNG), upload(u, SAMPLE_WAV)
            gen(u, image=img["name"], audio1=wav["name"])
            job = finish(u)[0]
            s.check("out of GPU memory comes back as advice",
                    job["status"] == "error" and "smaller size" in job["error"],
                    job.get("error"))
            jid = job["id"]
            log = requests.get(f"{u}/api/jobs/{jid}/log", timeout=10).json()
            s.check("the engine's own lines are there to read",
                    any("CUDA out of memory" in ln for ln in log["lines"]))
            s.equal("a failed render adds nothing to the gallery",
                    requests.get(u + "/api/clips", timeout=10).json(), [])

        # -- not set up ----------------------------------------------------------
        with studio(ws / "data4", ws / "empty-weights") as app:
            u = app.url
            st = requests.get(u + "/api/status", timeout=10).json()
            s.check("missing weights block rendering and are named",
                    not st["ready"] and "Wan2.1_VAE.pth" in st["missing_models"])
            img, wav = upload(u, SAMPLE_PNG), upload(u, SAMPLE_WAV)
            r = gen(u, image=img["name"], audio1=wav["name"])
            s.check("generate says what is missing instead of queueing",
                    r.status_code == 503 and "Models page" in r.json()["error"])
            deps = requests.get(u + "/api/deps", timeout=60).json()["items"]
            s.check("the dependency report names the weights",
                    any(i["id"] == "models" and i["state"] == "missing"
                        for i in deps))

            # -- downloads against the stand-in hub ----------------------------
            with hub() as hf:
                requests.post(u + "/api/hf/settings",
                              json={"endpoint": hf.url}, timeout=10)
                requests.post(hf.url + "/mock/mode", json={"cut_after": 200_000},
                              timeout=10)
                r = requests.post(u + "/api/hf/download", json={}, timeout=60)
                s.check("the set downloads, folders expanded into files",
                        r.ok and len(r.json()["tasks"]) >= 15, r.text[:200])
                wait_for(lambda: not [t for t in requests.get(
                    u + "/api/tasks", timeout=10).json() if t["state"] == "running"],
                    60)
                tasks = requests.get(u + "/api/tasks", timeout=10).json()
                cut = [t for t in tasks if t["state"] == "error"]
                s.check("a dropped connection fails that file only",
                        len(cut) == 1 and "FusionX" in cut[0]["title"],
                        str([(t["title"], t["state"]) for t in cut]))
                part = (ws / "empty-weights" / "MeiGen-MultiTalk" / "quant_models"
                        / "quant_model_int8_FusionX.safetensors.part")
                kept = part.stat().st_size if part.exists() else 0
                s.check("the part it got is kept (whole 64 kB chunks)",
                        0 < kept <= 200_000 and kept % 65536 == 0, str(kept))
                r = requests.post(u + "/api/hf/download", json={}, timeout=60)
                s.equal("asking again queues only what is missing",
                        len(r.json()["tasks"]), 1)
                wait_for(lambda: not [t for t in requests.get(
                    u + "/api/tasks", timeout=10).json() if t["state"] == "running"],
                    60)
                logs = requests.get(hf.url + "/mock/log", timeout=10).json()
                s.check("the retry resumed from where it stopped",
                        any(f"FusionX.safetensors from {kept}" in ln for ln in logs))
                done = (ws / "empty-weights" / "MeiGen-MultiTalk" / "quant_models"
                        / "quant_model_int8_FusionX.safetensors")
                s.check("the resumed file is whole", done.exists()
                        and done.stat().st_size == 300_000)
                s.check("wav2vec came from its PR branch",
                        any("chinese-wav2vec2-base@refs/pr/1/model.safetensors"
                            in ln for ln in logs))
                st = requests.get(u + "/api/status", timeout=10).json()
                s.check("with the set downloaded, the app is ready",
                        st["ready"], st.get("blocker"))
                local = requests.get(u + "/api/hf/local", timeout=10).json()
                s.check("the weights are listed on the Models page",
                        any(m["path"].endswith("Wan2.1_VAE.pth")
                            for m in local["models"]))
                r = requests.delete(u + "/api/hf/local",
                                    json={"path": "../data4/config.json"},
                                    timeout=10)
                s.equal("deleting outside the weights folder is refused",
                        r.status_code, 400)
                r = requests.delete(u + "/api/hf/local",
                                    json={"path": "Wan2.1-I2V-14B-480P/config.json"},
                                    timeout=10)
                s.check("deleting a weight works and the app notices",
                        r.ok and not requests.get(u + "/api/status",
                                                  timeout=10).json()["ready"])
    return s
