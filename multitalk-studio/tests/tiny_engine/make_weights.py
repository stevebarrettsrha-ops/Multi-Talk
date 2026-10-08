"""Build a weights folder for the tiny engine.

    <engine-venv python> tests/tiny_engine/make_weights.py <weights dir>

Every file the studio's weight check looks for is put in place, so the app
reads "ready". Only one of them is a real model: wav2vec2 at the real
chinese-wav2vec2-base shape (12 layers, 768 wide) with random weights,
because the real audio code loads and runs it. The rest are placeholders
the tiny engine never opens. Run it with the engine environment's Python:
it needs transformers.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import bootstrap  # noqa: E402


def main(wdir: Path) -> None:
    for item in bootstrap.model_set({"precision": "int8-fusionx",
                                     "want_tts": True}):
        target = bootstrap.model_path(wdir, item)
        if item["prefix"]:
            target.mkdir(parents=True, exist_ok=True)
            for n in (["af_heart.pt", "am_adam.pt"] if "voices" in item["path"]
                      else ["tokenizer.json"]):
                (target / n).write_bytes(b"placeholder")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"placeholder")

    from transformers import (Wav2Vec2Config, Wav2Vec2FeatureExtractor,
                              Wav2Vec2Model)
    w2v = wdir / bootstrap.WAV2VEC_DIR
    Wav2Vec2Model(Wav2Vec2Config()).save_pretrained(w2v, safe_serialization=True)
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
                             padding_value=0.0, do_normalize=True,
                             return_attention_mask=False).save_pretrained(w2v)
    print("weights ready in", wdir)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
