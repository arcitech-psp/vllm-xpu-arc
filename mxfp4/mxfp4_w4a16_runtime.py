"""Opt-in packed MXFP4 W4A16 operator for an isolated XPU candidate."""
import os
import torch

torch.ops.load_library(os.environ['VLLM_XPU_MXFP4_W4A16_LIBRARY'])

@torch.library.register_fake('vllm_xpu_mxfp4_woq::matmul')
def _fake(a, weight, scales):
    torch._check(a.dim() == 2)
    torch._check(weight.dim() == 2)
    torch._check(scales.dim() == 2)
    torch._check(a.shape[1] == weight.shape[1] * 2)
    torch._check(scales.shape[0] * 32 == a.shape[1])
    torch._check(scales.shape[1] == weight.shape[0])
    return a.new_empty((a.shape[0], weight.shape[0]))

def apply_w4a16(layer, x, bias=None):
    """Existing vLLM transposed parameter storage is retained unchanged."""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    weight = layer.weight.t()
    # This is a view back to original contiguous packed storage, not a copy.
    out = torch.ops.vllm_xpu_mxfp4_woq.matmul(x2, weight, layer.weight_scale)
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], weight.shape[0])
