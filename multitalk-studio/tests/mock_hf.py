"""A stand-in for huggingface.co: the tree API and resolve downloads with
Range support, so the download, resume and folder-expansion paths run for
real. The file names are the ones the real repos carry."""
import sys

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
LOG = []
MODE = {"cut_after": 0}

REPOS = {
    "MeiGen-AI/MeiGen-MultiTalk": {
        "quant_models/quant_model_int8_FusionX.safetensors": 300_000,
        "quant_models/quantization_map_int8_FusionX.json": 2_000,
        "quant_models/dit_model_int8.safetensors": 280_000,
        "quant_models/dit_model_map_int8.json": 2_000,
        "quant_models/t5_int8.safetensors": 120_000,
        "quant_models/t5_map_int8.json": 1_000,
        "multitalk.safetensors": 50_000,
        "README.md": 900,
    },
    "Wan-AI/Wan2.1-I2V-14B-480P": {
        "config.json": 300, "Wan2.1_VAE.pth": 40_000,
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": 90_000,
        "google/umt5-xxl/tokenizer.json": 5_000,
        "google/umt5-xxl/spiece.model": 4_000,
        "google/umt5-xxl/tokenizer_config.json": 300,
        "xlm-roberta-large/tokenizer.json": 5_000,
        "xlm-roberta-large/sentencepiece.bpe.model": 4_000,
        "diffusion_pytorch_model-00001-of-00007.safetensors": 999_999,
    },
    "TencentGameMate/chinese-wav2vec2-base": {
        "config.json": 300, "preprocessor_config.json": 200,
        "pytorch_model.bin": 30_000,
    },
    "TencentGameMate/chinese-wav2vec2-base@refs/pr/1": {
        "model.safetensors": 30_000,
    },
    "hexgrad/Kokoro-82M": {
        "config.json": 300, "kokoro-v1_0.pth": 20_000,
        "voices/af_heart.pt": 500, "voices/am_adam.pt": 500,
    },
}


def body(name: str, size: int) -> bytes:
    seed = name.encode()
    return (seed * (size // max(len(seed), 1) + 1))[:size]


@app.post("/mock/mode")
def mode():
    MODE.update(request.get_json(silent=True) or {})
    return jsonify(MODE)


@app.get("/mock/log")
def log():
    return jsonify(LOG)


@app.get("/api/models/<org>/<name>/tree/<path:rev>")
def tree(org, name, rev):
    key = f"{org}/{name}" + ("" if rev == "main" else f"@{rev}")
    LOG.append(f"tree {key}")
    if key not in REPOS:
        return jsonify({"error": "not found"}), 404
    return jsonify([{"type": "file", "path": p, "size": s}
                    for p, s in REPOS[key].items()])


@app.get("/<org>/<name>/resolve/<path:rest>")
def resolve(org, name, rest):
    # %2F in "refs%2Fpr%2F1" arrives decoded, so the revision is either the
    # first segment or refs/pr/<n>
    parts = rest.split("/")
    cut = 3 if parts[0] == "refs" else 1
    rev, path = "/".join(parts[:cut]), "/".join(parts[cut:])
    key = f"{org}/{name}" + ("" if rev == "main" else f"@{rev}")
    files = REPOS.get(key, {})
    if path not in files:
        return "not found", 404
    data = body(path, files[path])
    start = 0
    rng = request.headers.get("Range", "")
    if rng.startswith("bytes="):
        start = int(rng[6:].split("-")[0])
        if start >= len(data):
            return Response(status=416,
                            headers={"Content-Range": f"bytes */{len(data)}"})
    LOG.append(f"get {key}/{path} from {start}")
    chunk = data[start:]
    if MODE["cut_after"] and path.endswith("FusionX.safetensors") and start == 0:
        MODE["cut_after"], cut = 0, MODE["cut_after"]
        # a dropped connection: promise everything, deliver part of it
        def gen():
            yield chunk[:cut]
            raise ConnectionError("cut")
        return Response(gen(), status=200,
                        headers={"Content-Length": str(len(chunk))})
    status = 206 if start else 200
    headers = {"Content-Length": str(len(chunk))}
    if start:
        headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
    return Response(chunk, status=status, headers=headers)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(sys.argv[1]), threaded=True)
