"""B70 draft MTP INT4 helper with BF16-safe module boundaries.

The W4A16 kernel may use FP16 internally, but every custom draft linear casts
its result back to the input/model dtype.  Only draft predictor linears are
annotated; target modules and parameters are not traversed or modified.
"""
from __future__ import annotations

import os

import torch


def quantize_to_int4(weight: torch.Tensor, group_size: int = 128):
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts = []
    scale_parts = []
    for i in range(0, N, 4096):
        wc = weight[i : i + 4096].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        scale = wg.abs().amax(dim=-1) / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        qv = (q + 8).view(wc.shape[0], num_groups, group_size // 8, 8)
        parts.append(
            (qv << shifts).sum(dim=-1).to(torch.int32).reshape(wc.shape[0], K // 8)
        )
        scale_parts.append(scale.to(torch.float16))
    qweight = torch.cat(parts, dim=0).t()
    scales = torch.cat(scale_parts, dim=0).t().contiguous()
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def _collect_linears(predictor) -> list[tuple[str, torch.nn.Module]]:
    found: list[tuple[str, torch.nn.Module]] = [("fc", predictor.fc)]
    for li, layer in enumerate(predictor.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            found.append((f"layers.{li}.self_attn.qkv_proj", attn.qkv_proj))
            found.append((f"layers.{li}.self_attn.o_proj", attn.o_proj))
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            gate_up = getattr(mlp, "gate_up_proj", None)
            down = getattr(mlp, "down_proj", None)
            if gate_up is not None:
                found.append((f"layers.{li}.mlp.gate_up_proj", gate_up))
            if down is not None:
                found.append((f"layers.{li}.mlp.down_proj", down))
    return found


class _VllmXpuMTPInt4LinearMethod:
    def __init__(self, qweight, scales, qzeros, group_size):
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.group_size = group_size

    def create_weights(self, *args, **kwargs):
        pass

    def process_weights_after_loading(self, *args, **kwargs):
        pass

    def apply(self, layer, x, bias):
        flat = x.reshape(-1, x.shape[-1])
        model_dtype = x.dtype
        weight = getattr(layer, "weight", None)
        if weight is not None:
            model_dtype = weight.dtype
        kernel_x = flat if flat.dtype == torch.float16 else flat.to(torch.float16)
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            kernel_x, self.qweight, None, self.scales, self.qzeros,
            self.group_size, None,
        )
        # The custom output is a draft-only intermediate.  Restore the BF16
        # model dtype before it enters an unquantized draft MoE/router layer.
        return out.to(model_dtype).reshape(*x.shape[:-1], self.qweight.shape[1])


@torch.no_grad()
def build_draft_mtp_int4(model) -> None:
    if (os.environ.get("B70_DRAFT_MTP_INT4") != "1"
            and os.environ.get("VLLM_XPU_DRAFT_MTP_INT4") != "1"):
        return
    if getattr(model, "_vllm_xpu_mtp_int4_built", False):
        return
    predictor = getattr(model, "model", None)
    if predictor is None or not hasattr(predictor, "layers"):
        print("[vllm-xpu] draft MTP INT4: predictor unavailable; skip", flush=True)
        return
    linears = _collect_linears(predictor)
    print(f"[vllm-xpu] draft MTP INT4: quantizing {len(linears)} draft linears", flush=True)
    total_fp16 = 0
    total_int4 = 0
    for name, lin in linears:
        w = getattr(lin, "weight", None)
        if w is None:
            continue
        qweight, scales, qzeros, gs = quantize_to_int4(w.detach())
        lin._vllm_xpu_mtp_int4 = _VllmXpuMTPInt4LinearMethod(
            qweight, scales, qzeros, gs
        )
        lin.quant_method = lin._vllm_xpu_mtp_int4
        fp_bytes = w.numel() * w.element_size()
        int_bytes = qweight.numel() * qweight.element_size() + scales.numel() * scales.element_size()
        total_fp16 += fp_bytes
        total_int4 += int_bytes
        print(
            f"[vllm-xpu] draft MTP INT4: {name} shape={tuple(w.shape)} "
            f"dtype={w.dtype} -> {int_bytes/1e6:.0f} MB; BF16 output boundary",
            flush=True,
        )
    model._vllm_xpu_mtp_int4_built = True
    print(
        f"[vllm-xpu] draft MTP INT4: ready; {total_fp16/1e9:.2f} GB -> "
        f"{total_int4/1e9:.2f} GB INT4; target untouched",
        flush=True,
    )
