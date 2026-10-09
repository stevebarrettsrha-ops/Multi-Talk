"""Offline fixtures: no real model downloads, no user cache or settings touched."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from harness import Suite, fake_weights
import bootstrap
import engine
import manager


def snapshot(cache, repo, revision="main", commit="a" * 40):
    base = cache / ("models--" + repo.replace("/", "--"))
    ref = base / "refs" / revision
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit)
    dest = base / "snapshots" / commit
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def put(path, contents=b"valid fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def run():
    s = Suite("reuse")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache, weights = root / "hub", root / "weights"
        with patch.object(bootstrap, "hf_cache_roots", return_value=[cache]), \
                patch.object(bootstrap.requests, "get", side_effect=AssertionError("Network used")) as net:
            snap = snapshot(cache, "org/model")
            source = put(snap / "weights.safetensors")
            dest = weights / "weights.safetensors"
            progress = []
            bootstrap.download_file({}, "org/model", "weights.safetensors", dest,
                                    lambda *p: progress.append(p))
            s.equal("cached bytes adopted offline", dest.read_bytes(), source.read_bytes())
            s.equal("cache adoption reports completion", progress[-1][:2], (13, 13))
            s.check("cache bytes are retained", source.is_file())
            dest.unlink()
            s.check("deleting adopted model keeps the cache", source.is_file())
            put(dest, b"existing downloaded bytes")
            bootstrap.download_file({}, "org/model", "weights.safetensors", dest)
            s.equal("completed destination is retained", dest.read_bytes(), b"existing downloaded bytes")
            s.equal("reuse and skip operations use zero network requests", net.call_count, 0)
            s.check("a similarly named repo is not substituted",
                    bootstrap.cached_file("other/model", "weights.safetensors") is None)
            s.check("a different revision is not substituted",
                    bootstrap.cached_file("org/model", "weights.safetensors", "refs/pr/1") is None)
            pr = snapshot(cache, "org/model", "refs/pr/1", "b" * 40)
            put(pr / "weights.safetensors", b"correct PR weights")
            s.equal("requested PR branch is resolved independently",
                    bootstrap.cached_file("org/model", "weights.safetensors", "refs/pr/1").read_bytes(),
                    b"correct PR weights")
            s.check("known-size truncated cache file is rejected",
                    bootstrap.cached_file("org/model", "weights.safetensors", expected_size=100) is None)
            put(snap / "pointer.pth", b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
            put(snap / "empty.pth", b"")
            part = put(snap / "chunk.incomplete")
            (snap / "broken.pth").symlink_to(part)
            for name in ("pointer.pth", "empty.pth", "broken.pth"):
                s.check(f"unfinished cache entry {name} rejected",
                        bootstrap.cached_file("org/model", name) is None)
            copied = weights / "copied.pth"
            with patch.object(bootstrap.os, "link", side_effect=OSError("different drives")):
                bootstrap.reuse_cache(source, copied)
            s.equal("different-drive copy reuses bytes offline", copied.read_bytes(), source.read_bytes())
            cancel = weights / "cancel.pth"
            bootstrap.reuse_cache(source, cancel, should_cancel=lambda: True)
            s.check("cancelled cache adoption creates no final model", not cancel.exists())

            # Actual model names/layout, with tiny stand-in bytes/estimates.
            real_set = bootstrap.model_set
            def tiny_set(cfg):
                return [{**m, "size": 13} for m in real_set(cfg)]
            cfg = dict(bootstrap.DEFAULT_CONFIG, weights_dir=str(root / "set"), want_tts=False)
            with patch.object(bootstrap, "model_set", side_effect=tiny_set):
                for item in tiny_set(cfg):
                    snap = snapshot(cache, item["repo"], item["revision"])
                    path = snap / item["path"]
                    if item["prefix"]:
                        put(path / "tokenizer.json")
                        put(path / "tokenizer_config.json")
                    else:
                        put(path)
                wan = snapshot(cache, bootstrap.WAN_REPO)
                put(wan / "diffusion_pytorch_model-00001-of-00007.safetensors")
                files = bootstrap.expand(cfg, tiny_set(cfg))
                for item in files:
                    bootstrap.download_file(cfg, item["repo"], item["path"],
                                            bootstrap.model_path(bootstrap.weights_dir(cfg), item),
                                            revision=item["revision"], expected_size=item["known_size"])
                s.equal("full cached required set becomes ready offline",
                        bootstrap.missing_models(bootstrap.weights_dir(cfg), cfg), [])
                s.equal("set download skips all reused models", manager.download_set(cfg), [])
                s.equal("no tree or file requests with a full cache", net.call_count, 0)
                s.check("unneeded bf16 shards stay out of the install",
                        not list(bootstrap.weights_dir(cfg).rglob("diffusion_pytorch_model*")))
                job = engine.normalize({"image": "a.png", "audio1": "a.wav"}, cfg)
                argv = engine.argv(cfg, job, root / "input.json", root / "audio", root / "result")
                s.equal("engine points at the adopted Wan directory",
                        argv[argv.index("--ckpt_dir") + 1],
                        engine.as_path(bootstrap.weights_dir(cfg) / bootstrap.WAN_DIR))

    # A disconnected external model drive is not an invitation to switch
    # back to the empty weights folder bundled with the engine.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eng = root / "engine"
        put(eng / "generate_multitalk.py")
        (eng / "weights").mkdir()
        external = root / "external-drive" / "weights"
        unrelated = root / "empty-suffix-match" / "weights"
        unrelated.mkdir(parents=True)
        cfg = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(eng),
                   weights_dir=str(external), want_tts=False)
        with patch.object(bootstrap, "rebase_path", return_value=unrelated), \
                patch.object(bootstrap, "find_engine_installs") as scan, \
                patch.object(bootstrap, "heal_weights", wraps=bootstrap.heal_weights) as heal:
            bootstrap.verify_locations(cfg)
            s.equal("missing separate weights drive keeps its saved location",
                    cfg["weights_dir"], str(external))
            s.equal("an unrelated empty suffix does not replace external weights",
                    bootstrap.weights_dir(cfg), external)
            cfg = json.loads(json.dumps(cfg))
            bootstrap.verify_locations(cfg)
            bootstrap.verify_locations(cfg)
            s.equal("unchanged missing drive does not repeat relocation attempts",
                    heal.call_count, 1)
            s.equal("valid engine does not trigger a drive search", scan.call_count, 0)
            # Both setup and Download set resolve through this same weights_dir.
            item = bootstrap.model_set(cfg)[0]
            with patch.object(bootstrap, "expand", return_value=[item]), \
                    patch.object(manager, "download_item", return_value=None) as download:
                manager.download_set(cfg)
            saved_cfg, planned = download.call_args.args
            s.check("download destination stays on the explicit external root",
                    bootstrap.model_path(bootstrap.weights_dir(saved_cfg), planned)
                    .is_relative_to(external))
            s.check("failed checks did not create either missing destination",
                    not external.exists() and list((eng / "weights").iterdir()) == [])
            fake_weights(external, tts=False)
            bootstrap.verify_locations(cfg)
            s.equal("reappearing external drive resumes at the saved path",
                    cfg["weights_dir"], str(external))
            with patch.object(bootstrap.requests, "get", side_effect=AssertionError("Network used")) as net:
                s.equal("returned drive's downloaded weights are reused", manager.download_set(cfg), [])
                s.equal("returned weights require no requests", net.call_count, 0)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_eng, new_eng = root / "old-engine", root / "new-engine"
        put(new_eng / "generate_multitalk.py")
        (new_eng / "weights").mkdir()
        external = root / "missing-external" / "weights"
        cfg = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(old_eng),
                   weights_dir=str(old_eng / "weights"))
        with patch.dict(bootstrap.DEFAULT_CONFIG, {"engine_dir": str(new_eng)}), \
                patch.object(bootstrap, "rebase_path", return_value=None):
            bootstrap.verify_locations(cfg, search=False)
            s.equal("quick relocation keeps engine-relative default weights together",
                    bootstrap.weights_dir(cfg), new_eng / "weights")
            separate = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(old_eng),
                            weights_dir=str(external))
            bootstrap.verify_locations(separate, search=False)
            s.equal("quick engine relocation preserves separate missing weights",
                    bootstrap.weights_dir(separate), external)
        with patch.dict(bootstrap.DEFAULT_CONFIG, {"engine_dir": str(root / "absent-default")}), \
                patch.object(bootstrap, "rebase_path", return_value=None), \
                patch.object(bootstrap, "find_engine_installs", return_value=[new_eng]) as scan:
            separate = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(old_eng),
                            weights_dir=str(external))
            bootstrap.verify_locations(separate)
            s.equal("full search relocates the engine", bootstrap.engine_dir(separate), new_eng)
            s.equal("full search preserves separate missing weights", bootstrap.weights_dir(separate), external)
            bootstrap.verify_locations(separate)
            s.equal("full engine search is not repeated for disconnected weights", scan.call_count, 1)
            attached = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(old_eng),
                            weights_dir=str(old_eng / "weights"))
            bootstrap.verify_locations(attached)
            s.equal("full search also moves engine-relative default weights",
                    bootstrap.weights_dir(attached), new_eng / "weights")
        relocated = root / "validated-external-move"
        fake_weights(relocated, tts=False)
        separate = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(new_eng),
                        weights_dir=str(external), want_tts=False)
        with patch.object(bootstrap, "rebase_path", return_value=relocated):
            bootstrap.verify_locations(separate)
        s.equal("complete required model set validates a separate folder relocation",
                bootstrap.weights_dir(separate), relocated)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(root / "gone"))
        with patch.object(bootstrap, "heal_paths", return_value=[]) as heal, \
                patch.object(bootstrap, "find_engine_installs", return_value=[]) as scan:
            bootstrap.verify_locations(cfg)
            s.equal("initial missing engine gets one search", scan.call_count, 1)
            cfg = json.loads(json.dumps(cfg))
            for _ in range(4):
                bootstrap.verify_locations(cfg, search=False)
                bootstrap.verify_locations(cfg, search=True)
            s.equal("failed search persists across restart and polling", scan.call_count, 1)
            s.equal("same failed state does not repeat tail recovery", heal.call_count, 1)
            bootstrap.verify_locations(cfg, force=True)
            s.equal("explicit recheck allows one new search", scan.call_count, 2)
            cfg["engine_dir"] = str(root / "different")
            bootstrap.verify_locations(cfg)
            s.equal("new missing location gets one attempt", scan.call_count, 3)
            eng = root / "found"
            put(eng / "generate_multitalk.py")
            cfg["engine_dir"] = str(eng)
            for _ in range(4):
                bootstrap.verify_locations(cfg)
            s.equal("valid saved engine never walks drives", scan.call_count, 3)
            (eng / "generate_multitalk.py").unlink()
            bootstrap.verify_locations(cfg)
            bootstrap.verify_locations(cfg)
            s.equal("verified location failing gets exactly one search", scan.call_count, 4)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Importing the server must not touch a real user's config or search.
        with patch.dict(os.environ, {"MULTITALK_STUDIO_NO_SEARCH": "1"}), \
                patch.object(bootstrap, "DATA_DIR", root / "data"), \
                patch.object(bootstrap, "CONFIG_PATH", root / "data/config.json"):
            import server
        class ImmediateThread:
            def __init__(self, target, args=(), **kwargs):
                self.target, self.args = target, args
            def start(self):
                self.target(*self.args)
        cfg = dict(bootstrap.DEFAULT_CONFIG, engine_dir=str(root / "missing"))
        with patch.dict(os.environ, {"MULTITALK_STUDIO_NO_SEARCH": "0"}), \
                patch.object(server, "cfg", cfg), \
                patch.object(server, "save_config") as save, \
                patch.object(server, "_say"), \
                patch.object(bootstrap, "heal_paths", return_value=[]), \
                patch.object(bootstrap, "find_engine_installs", return_value=[]) as scan, \
                patch.object(manager, "dependencies", return_value=[]), \
                patch.object(server.threading, "Thread", ImmediateThread):
            client = server.app.test_client()
            client.get("/api/deps")
            s.equal("first dependency request attempts recovery", scan.call_count, 1)
            client.get("/api/deps?fresh=1")
            client.get("/api/deps")
            server._heal(search=True)  # boot follows the same remembered state
            s.equal("package refresh, polling and boot do not rescan", scan.call_count, 1)
            s.check("failed recovery metadata is saved", save.called)
            client.get("/api/deps?fresh=1&relocate=1")
            s.equal("only explicit relocation request permits another scan", scan.call_count, 2)
    return s
