"""CPU checks for the low-VRAM engine patches (MultiTalk Studio).

    python tests/test_lowvram_patches.py

Random weights, small shapes: these prove the patched code paths compute the
same thing as the originals, not that a real checkpoint looks good.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


failures = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(what)


torch.manual_seed(0)

# ---------------------------------------------------------------- VAE ----
vae_mod = load("vae_mod", "wan/modules/vae.py")
cfg = dict(dim=8, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
           attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
model = vae_mod.WanVAE_(**cfg).eval()
scale = [torch.zeros(4), torch.ones(4)]

with torch.no_grad():
    z = torch.randn(1, 4, 3, 24, 40)          # latent: 3 frames, 24x40
    plain = model.decode(z, scale)
    model.tile_size, model.tile_overlap = 16, 6
    tiled = model.decode(z, scale)
    model.tile_size = 0
check("tiled decode keeps the output shape",
      plain.shape == tiled.shape, f"{tuple(plain.shape)} vs {tuple(tiled.shape)}")
err = (plain - tiled).abs().mean().item() / (plain.abs().mean().item() + 1e-8)
check("tiled decode stays close to the plain decode (random weights)",
      err < 0.15, f"mean relative error {err:.4f}")

with torch.no_grad():
    x = torch.randn(1, 3, 9, 192, 256)        # 9 frames, 192x256 pixels
    plain_e = model.encode(x, scale)
    model.tile_size, model.tile_overlap = 16, 6
    tiled_e = model.encode(x, scale)
    model.tile_size = 0
check("tiled encode keeps the latent shape",
      plain_e.shape == tiled_e.shape, f"{tuple(plain_e.shape)} vs {tuple(tiled_e.shape)}")
err = (plain_e - tiled_e).abs().mean().item() / (plain_e.abs().mean().item() + 1e-8)
check("tiled encode stays close to the plain encode (random weights)",
      err < 0.15, f"mean relative error {err:.4f}")

with torch.no_grad():
    model.tile_size, model.tile_overlap = 64, 8
    big = model.decode(z, scale)              # tile larger than the latent
    model.tile_size = 0
check("a tile larger than the picture is the plain decode exactly",
      torch.equal(big, plain))

# MultiTalk Studio: the host accumulation path must preserve the same blend.
with torch.no_grad():
    model.tile_size, model.tile_overlap = 16, 6
    normal = model.decode(z, scale)
    model.decode_output_device = "cpu"
    host = model.decode(z, scale)
    model.decode_output_device = None
check("host accumulation preserves tiled pixel values", torch.equal(normal, host))
check("the offloaded tiled canvas is in host memory", host.device.type == "cpu")

ones = vae_mod.WanVAE_._ramp_mask(10, 12, 3, 3, True, True, True, True,
                                  "cpu", torch.float32)
check("blend ramps never reach zero", ones.min().item() > 0)
starts = vae_mod.WanVAE_._tile_starts(60, 32, 8)
check("tiles cover the whole edge", starts[0] == 0 and starts[-1] + 32 == 60,
      str(starts))

# ---------------------------------------------------------- attention ----
# attention.py imports relative modules, so load it as part of the package
# with xfuser and xformers absent
for missing in ("xfuser", "xformers", "flash_attn", "flash_attn_interface"):
    sys.modules[missing] = None
# The real package __init__s pull in T5, whose defaults touch CUDA at import
# time; stub the packages so only the modules under test load.
for pkg in ("wan", "wan.modules", "wan.utils"):
    stub = types.ModuleType(pkg)
    stub.__path__ = [str(ROOT / pkg.replace(".", "/"))]
    sys.modules[pkg] = stub
import importlib  # noqa: E402
att = importlib.import_module("wan.modules.attention")

check("the module imports without xfuser, xformers or flash-attn",
      att.xformers is None and not att.FLASH_ATTN_2_AVAILABLE)

B, L, N, C = 2, 10, 4, 16
q = torch.randn(B, L, N, C, dtype=torch.bfloat16)
k = torch.randn(B, 7, N, C, dtype=torch.bfloat16)
v = torch.randn(B, 7, N, C, dtype=torch.bfloat16)
k_lens = torch.tensor([7, 4])
out = att.flash_attention(q, k, v, k_lens=k_lens)


def reference(qb, kb, vb):
    w = torch.softmax((qb.float() @ kb.float().transpose(-1, -2))
                      / math.sqrt(C), dim=-1)
    return w @ vb.float()


ref = torch.stack([
    reference(q[i].transpose(0, 1), k[i, :k_lens[i]].transpose(0, 1),
              v[i, :k_lens[i]].transpose(0, 1)).transpose(0, 1)
    for i in range(B)])
check("SDPA fallback matches softmax attention with key lengths",
      out.shape == (B, L, N, C) and
      (out.float() - ref).abs().max().item() < 0.05,
      f"max error {(out.float() - ref).abs().max().item():.4f}")

q_lens = torch.tensor([10, 6])
out2 = att.flash_attention(q, k, v, q_lens=q_lens)
check("query padding past q_lens is zero, like the flash path",
      out2[1, 6:].abs().max().item() == 0)

qa = torch.randn(1, 12, 4, 16)
ka = torch.randn(1, 5, 4, 16)
va = torch.randn(1, 5, 4, 16)
mea = att._memory_efficient_attention(qa, ka, va)
ref2 = reference(qa[0].transpose(0, 1), ka[0].transpose(0, 1),
                 va[0].transpose(0, 1)).transpose(0, 1)
check("audio attention without xformers matches softmax attention",
      (mea[0] - ref2).abs().max().item() < 1e-4)

# ----------------------------------------------------------- buckets ----
mu = load("mu", "wan/utils/multitalk_utils.py")
check("every size has a bucket table",
      set(mu.BUCKET_TABLES) == {"multitalk-240", "multitalk-360",
                                "multitalk-480", "multitalk-720"})
check("the small buckets keep the same aspect keys as 480",
      set(mu.ASPECT_RATIO_480) == set(mu.ASPECT_RATIO_627)
      == set(mu.ASPECT_RATIO_320))
check("every small bucket is a multiple of 32 (VAE x patch stride)",
      all(v % 32 == 0 for t in (mu.ASPECT_RATIO_480, mu.ASPECT_RATIO_320)
          for (hw, _) in t.values() for v in hw))
check("the small buckets keep each key's shape (h/w within 10%)",
      all(abs((hw[0] / hw[1]) / (big[0][0] / big[0][1]) - 1) < 0.1
          for t in (mu.ASPECT_RATIO_480, mu.ASPECT_RATIO_320)
          for key, (hw, _) in t.items()
          for big in [mu.ASPECT_RATIO_627[key]]))
check("ffmpeg_exe finds something to run", bool(mu.ffmpeg_exe()))

# ------------------------------------------------- INT8 weight loading ----
import tempfile
from safetensors.torch import save_file, load_file
from optimum.quanto import freeze, qint8, quantize, quantization_map, requantize

lm = load("lm", "wan/utils/lowmem_load.py")
tmp = Path(tempfile.mkdtemp())
sd = {"f32": torch.randn(3, 5), "bf16": torch.randn(7, 2).bfloat16(),
      "i8": torch.randint(-128, 127, (9, 3), dtype=torch.int8),
      "scalar": torch.tensor(2.5), "empty": torch.zeros(0, 4)}
save_file(sd, tmp / "a.safetensors", metadata={"format": "pt"})
ref = load_file(tmp / "a.safetensors")
for mapped in (False, True):
    got = lm.load_safetensors(tmp / "a.safetensors", mapped=mapped)
    check(f"the reader (mapped={mapped}) gives what safetensors' load_file gives",
          ref.keys() == got.keys() and all(
              ref[k].dtype == got[k].dtype and ref[k].shape == got[k].shape
              and torch.equal(ref[k].reshape(-1).view(torch.uint8),
                              got[k].reshape(-1).view(torch.uint8)) for k in ref))
(tmp / "cut.safetensors").write_bytes((tmp / "a.safetensors").read_bytes()[:-8])
for mapped in (False, True):
    try:
        lm.load_safetensors(tmp / "cut.safetensors", mapped=mapped)
        check(f"a truncated file is reported (mapped={mapped})", False)
    except OSError as e:
        check(f"a truncated file is reported (mapped={mapped})", "truncated" in str(e))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 128)
        self.n = torch.nn.LayerNorm(128)
        self.b = torch.nn.Linear(128, 32)

    def forward(self, x):
        return self.b(self.n(self.a(x)))


q = Tiny().to(torch.bfloat16)
quantize(q, weights=qint8)
freeze(q)
save_file(q.state_dict(), tmp / "q.safetensors")
qmap = quantization_map(q)
x = torch.randn(5, 64, dtype=torch.bfloat16)
for meta_dtype in (torch.float32, torch.bfloat16):  # the DiT's, the T5's
    with torch.device("meta"):
        up = Tiny().to(meta_dtype)
        mine = Tiny().to(meta_dtype)
    requantize(up, load_file(tmp / "q.safetensors"), qmap, device="cpu")
    lm.requantize_in_place(mine, lm.load_safetensors(tmp / "q.safetensors",
                                                     mapped=True), qmap)
    same = all(n1 == n2 and p1.dtype == p2.dtype and type(p1.data) is type(p2.data)
               and p2.device.type == "cpu"
               for (n1, p1), (n2, p2) in zip(up.named_parameters(),
                                             mine.named_parameters()))
    w = mine.a.weight.data
    check(f"the {meta_dtype} model's int8 weights stay the file's mapped pages",
          w._data.untyped_storage().nbytes() > w._data.numel())
    check(f"requantize_in_place matches requantize ({meta_dtype} model)",
          same and torch.equal(up.bfloat16()(x), mine.bfloat16()(x)))

print()
print("FAILED: " + ", ".join(failures) if failures else "all passed")
sys.exit(1 if failures else 0)
