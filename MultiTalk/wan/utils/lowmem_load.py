"""Loading the INT8 weights with the file held in RAM once (MultiTalk Studio).

Windows reserves ("commits") every allocation against RAM + page file up
front, so on a 32 GB PC two habits of the upstream loading code ran out
before a render started:

* safetensors' load_file() maps the file copy-on-write. Windows charges the
  whole mapping to the commit limit, then the tensors are copied out of it:
  the 16.5 GB DiT file counted twice ("The paging file is too small for this
  operation to complete. (os error 1455)").
* optimum-quanto's requantize() builds the whole model empty on the CPU in
  the meta model's dtype (float32 for the DiT, four times the INT8 file) and
  then loads the state dict into it (exit code 3221225477, 0xC0000005).

load_safetensors() reads each tensor straight from the file into its own
buffer, and requantize_in_place() makes those buffers the model's.
"""
import json
import struct

import torch

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool,
}
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _attr):
        _DTYPES[_name] = getattr(torch, _attr)


def load_safetensors(path):
    """safetensors.torch.load_file() without the memory map: same dict of
    CPU tensors, but nothing beyond the tensors themselves is committed."""
    out = {}
    with open(path, "rb", buffering=0) as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        base = 8 + header_len
        for name, info in sorted(header.items(),
                                 key=lambda kv: kv[1]["data_offsets"][0]):
            start, end = info["data_offsets"]
            buf = torch.empty(end - start, dtype=torch.uint8)
            view = memoryview(buf.numpy())
            f.seek(base + start)
            got = 0
            while got < len(view):
                n = f.readinto(view[got:])
                if not n:
                    raise OSError(f"{path} is truncated at tensor {name} "
                                  "(download it again on the Models page)")
                got += n
            out[name] = buf.view(_DTYPES[info["dtype"]]).reshape(info["shape"])
    return out


def requantize_in_place(model, state_dict, quantization_map):
    """optimum-quanto's requantize() for a model built on the meta device,
    without allocating the model a second time: the state dict's tensors
    become the model's (assign=True). Tensors that are not quantized end in
    the dtype the model declared, as upstream's copy into it gave them;
    anything the file does not carry is materialised empty on the CPU, as
    upstream does."""
    from optimum.quanto.quantize import _quantize_submodule

    for name, m in model.named_modules():
        qconfig = quantization_map.get(name)
        if qconfig is not None:
            weights = None if qconfig["weights"] == "none" else qconfig["weights"]
            activations = (None if qconfig["activations"] == "none"
                           else qconfig["activations"])
            _quantize_submodule(model, name, m, weights=weights,
                                activations=activations)
    declared = {(id(m), n): p.dtype for m in model.modules()
                for n, p in m.named_parameters(recurse=False)}
    model.load_state_dict(state_dict, strict=False, assign=True)
    for m in model.modules():
        for name, param in list(m.named_parameters(recurse=False)):
            t = param.data
            if t.device.type == "meta":
                t = torch.empty_like(t, device="cpu")
            want = declared.get((id(m), name), t.dtype)
            if type(t) is torch.Tensor and t.dtype != want:
                t = t.to(want)
            if t is not param.data:
                setattr(m, name, torch.nn.Parameter(t, requires_grad=False))
        for name, buf in list(m.named_buffers(recurse=False)):
            if buf is not None and buf.device.type == "meta":
                m._buffers[name] = torch.empty_like(buf, device="cpu")
