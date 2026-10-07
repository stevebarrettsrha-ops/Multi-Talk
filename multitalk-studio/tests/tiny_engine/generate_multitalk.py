"""MultiTalk's own generate_multitalk.py, run on the CPU with tiny random
weights — the end-to-end check this sandbox can do without a GPU or the
29 GB of real weights.

    python tests/tiny_engine/generate_multitalk.py <the studio's command line>

What is REAL here, unchanged from ../../../MultiTalk:
  - generate_multitalk.py: argument parsing, the input JSON, the audio
    preparation for one or two speakers (loading, loudness, "take turns" /
    "at once" mixing, padding), wav2vec2 embedding, the TTS route, and
    save_video_ffmpeg muxing the voices into the MP4
  - wan/multitalk.py: MultiTalkPipeline.generate() — size buckets, the
    two-person face masks, the audio windows per clip, streaming clip after
    clip with motion frames, CFG, the flow-matching steps, colour correction
  - the MultiTalk DiT class (WanModel with its audio adapter and audio
    cross-attention), at a tiny size with random weights
  - the Wan VAE, at its real size with random weights, tiled as on 8 GB
  - VRAM management wrapping, attention fallbacks (no flash-attn here)

What is STUBBED, because loading it is the only thing it would prove and
this machine has neither the memory nor the download: the umT5-XXL text
encoder and the CLIP image encoder. They return random features of the
real shapes (4096-wide text tokens, 257 x 1280 image tokens).

So the video that comes out is noise — random weights draw noise — but every
frame of it went through the real pipeline, and its soundtrack is the real
two-voice mix. A picture that looks like the people in the photo needs the
real weights, on a GPU: that is what the studio's engine self-test checks.

The wav2vec2 folder must hold a real-shaped model (random weights are fine);
everything else under --ckpt_dir / --quant_dir is ignored.
"""
import importlib.util
import os
import sys
from pathlib import Path

ENGINE = Path(os.environ.get("TINY_ENGINE_SOURCE")
              or Path(__file__).resolve().parents[3] / "MultiTalk")
sys.path.insert(0, str(ENGINE))

import torch  # noqa: E402

# no GPU, no driver: the two CUDA calls the pipeline makes unconditionally
# (between steps, to free memory) become no-ops; everything else already
# follows the device it is given
torch.cuda.ipc_collect = lambda: None
torch.cuda.synchronize = lambda *a, **k: None
torch.set_num_threads(max(1, os.cpu_count() or 1))
# xformers and flash-attn have no CPU kernels; hidden, the engine takes its
# PyTorch-attention fallback (the one a Windows install without them uses)
for _gpu_only in ("xformers", "xformers.ops", "flash_attn", "flash_attn_interface",
                  "sageattention"):
    sys.modules[_gpu_only] = None

spec = importlib.util.spec_from_file_location(
    "generate_multitalk", ENGINE / "generate_multitalk.py")
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)

import wan  # noqa: E402
from wan import multitalk as wm  # noqa: E402
from wan.modules.multitalk_model import WanModel  # noqa: E402
from wan.modules.vae import WanVAE, WanVAE_  # noqa: E402


class StubText:
    """umT5-XXL's interface: .model, and a call that returns one
    [tokens, 4096] tensor per prompt."""

    def __init__(self):
        self.model = torch.nn.Identity()

    def __call__(self, texts, device):
        out = []
        for t in texts:
            g = torch.Generator().manual_seed(abs(hash(t)) % (2 ** 31))
            n = max(4, min(len(t.split()), 512))
            # float32 like the tiny model (see TinyPipeline): the model casts
            # its input to the text features' dtype
            out.append(torch.randn(n, 4096, generator=g).to(device))
        return out


class StubClip:
    """CLIP ViT-H's interface: .model, and .visual(image) -> [1, 257, 1280]."""

    def __init__(self):
        self.model = torch.nn.Identity()

    def visual(self, videos):
        g = torch.Generator().manual_seed(1)
        return torch.randn(1, 257, 1280, generator=g)


def real_vae(device):
    """The Wan VAE exactly as WanVAE.__init__ builds it, minus the
    checkpoint: same architecture, same latent scaling, random weights."""
    vae = object.__new__(WanVAE)
    vae.dtype, vae.device = torch.float, device
    mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    vae.mean = torch.tensor(mean, device=device)
    vae.std = torch.tensor(std, device=device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    torch.manual_seed(0)
    vae.model = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
                        num_res_blocks=2, attn_scales=[],
                        temperal_downsample=[False, True, True],
                        dropout=0.0).eval().requires_grad_(False).to(device)
    return vae


class TinyPipeline(wm.MultiTalkPipeline):
    """MultiTalkPipeline with its loading replaced; generate() is the real one."""

    def __init__(self, config, checkpoint_dir, quant_dir=None, device_id=0,
                 rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False,
                 t5_cpu=False, init_on_cpu=True, num_timesteps=1000,
                 use_timestep_transform=True, lora_dir=None, lora_scales=None,
                 quant=None):
        print("[tiny-engine] MultiTalk pipeline on the CPU with tiny random "
              "weights (T5 and CLIP stubbed) — see tests/tiny_engine", flush=True)
        self.device = torch.device("cpu")
        self.config, self.rank, self.use_usp = config, rank, False
        self.t5_cpu = True
        self.num_train_timesteps = config.num_train_timesteps
        # float32 throughout. Upstream mixes bf16 weights with float32 maths
        # inside torch.cuda.amp.autocast blocks, which cast both sides on a
        # GPU and do nothing at all for CPU tensors — so bf16 here fails on
        # dtypes a GPU never sees. The code path is the same either way.
        self.param_dtype = torch.float32
        self.vae_stride, self.patch_size = config.vae_stride, config.patch_size
        self.text_encoder, self.clip = StubText(), StubClip()
        self.vae = real_vae(self.device)
        torch.manual_seed(0)
        self.model = WanModel(dim=128, ffn_dim=256, num_heads=4, num_layers=2,
                              text_dim=4096, freq_dim=256, in_dim=36,
                              out_dim=16).eval().requires_grad_(False)
        self.sp_size = 1
        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.cpu_offload = False
        self.model_names = ["model"]
        self.vram_management = False


wan.MultiTalkPipeline = TinyPipeline

if __name__ == "__main__":
    gm.generate(gm._parse_args())
