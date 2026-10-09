"""MultiTalk Studio: real generate-loop lifecycle checks without checkpoints.

Tiny component doubles record live tensor references at the decode boundary.
The pipeline's clip loop, guidance, motion continuation and model offloading
run unchanged on CPU. This does not measure CUDA memory or visual quality.
"""
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wan import multitalk as pipeline_module
from wan.modules.multitalk_model import WanModel
from wan.utils import multitalk_utils


class WrappedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded = False
        self.offloads = 0

    def onload(self):
        self.loaded = True

    def offload(self):
        self.loaded = False
        self.offloads += 1


class RecordingModel(torch.nn.Module):
    teacache_init = WanModel.teacache_init
    clear_teacache = WanModel.clear_teacache
    disable_teacache = WanModel.disable_teacache

    def __init__(self):
        super().__init__()
        self.vram_management_enabled = True
        self.layer = WrappedLayer()
        self.references = []

    def forward(self, x, t, **kwargs):
        self.references.extend(weakref.ref(kwargs[key]) for key in
                               ("y", "audio", "clip_fea", "ref_target_masks"))
        if self.enable_teacache:
            self.previous_residual_cond = torch.zeros(2, 2)
            self.references.append(weakref.ref(self.previous_residual_cond))
        return [torch.zeros_like(x[0])]


class PipelineLifecycle(unittest.TestCase):
    def run_pipeline(self, *, text_scale=1, apg=False, streaming=False,
                     teacache=False):
        model = RecordingModel()
        pipe = object.__new__(pipeline_module.MultiTalkPipeline)
        pipe.device = torch.device("cpu")
        pipe.rank, pipe.sp_size = 0, 1
        pipe.t5_cpu = True
        pipe.param_dtype = torch.float32
        pipe.vae_stride, pipe.patch_size = (4, 8, 8), (1, 2, 2)
        pipe.sample_neg_prompt = "negative"
        pipe.num_timesteps, pipe.use_timestep_transform = 1000, False
        pipe.cpu_offload = pipe.vram_management = True
        pipe.model_names, pipe.model = ["model"], model
        pipe.text_encoder = lambda texts, device: [torch.zeros(2, 8) for _ in texts]
        pipe.clip = types.SimpleNamespace(model=torch.nn.Identity(),
                                         visual=lambda x: torch.zeros(1, 2, 8))
        decodes = []

        def decode(zs):
            self.assertFalse(model.layer.loaded, "DiT wrappers overlap VAE decode")
            self.assertTrue(all(ref() is None for ref in model.references),
                            "sampling tensor is still live at VAE decode")
            decodes.append(True)
            model.references.clear()
            return [torch.zeros(3, 5, 32, 32)]

        pipe.vae = types.SimpleNamespace(
            model=types.SimpleNamespace(),
            encode=lambda x: [torch.zeros(16, 2, 4, 4)], decode=decode)
        args = types.SimpleNamespace(use_teacache=teacache, use_apg=apg,
                                     teacache_thresh=0.3, size="multitalk-360",
                                     apg_momentum=-0.75, apg_norm_threshold=55,
                                     vae_tile=16, vae_tile_overlap=8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (32, 32)).save(root / "image.png")
            torch.save(torch.zeros(13, 2, 4), root / "audio.pt")
            data = {"prompt": "test", "cond_image": str(root / "image.png"),
                    "cond_audio": {"person1": str(root / "audio.pt")}}
            with patch.object(pipeline_module, "torch_gc"), \
                 patch.object(torch.cuda, "synchronize"), \
                 patch.dict(multitalk_utils.BUCKET_TABLES,
                            {"test": {1.0: ((32, 32), 1)}}):
                output = pipe.generate(data, size_buckget="test", motion_frame=1,
                                       frame_num=5, sampling_steps=2,
                                       max_frames_num=9 if streaming else 5,
                                       text_guide_scale=text_scale, audio_guide_scale=1,
                                       progress=False, extra_args=args)
        self.assertEqual(len(decodes), 2 if streaming else 1)
        self.assertEqual(model.layer.offloads, len(decodes))
        self.assertEqual(tuple(output.shape), (3, 9 if streaming else 5, 32, 32))

    def test_two_pass_guidance_releases_before_decode(self):
        self.run_pipeline()

    def test_three_pass_guidance_releases_before_decode(self):
        self.run_pipeline(text_scale=5)

    def test_apg_and_motion_continuation_release_before_decode(self):
        self.run_pipeline(apg=True, streaming=True)

    def test_three_pass_apg_releases_before_decode(self):
        self.run_pipeline(text_scale=5, apg=True)

    def test_teacache_apg_and_three_pass_continuation(self):
        self.run_pipeline(text_scale=5, apg=True, streaming=True, teacache=True)

    def test_teacache_release_clears_real_cache_attribute_names(self):
        model = RecordingModel()
        refs = []
        for branch in ("cond", "drop_text", "uncond"):
            for prefix in ("previous_e0_", "previous_residual_"):
                tensor = torch.ones(2, 2)
                refs.append(weakref.ref(tensor))
                setattr(model, prefix + branch, tensor)
        del tensor
        model.cnt = 8
        model.clear_teacache()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(model.cnt, 0)


if __name__ == "__main__":
    unittest.main()
