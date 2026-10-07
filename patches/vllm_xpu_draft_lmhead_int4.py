"""B70 draft LM-head INT4 helper with BF16 module-boundary hygiene.

The XPU W4A16 kernel uses an FP16 activation path internally.  Its result is
immediately converted back to the draft model dtype before it reaches any
BF16 draft module; the target model is never modified.
"""
from __future__ import annotations

import os

import torch


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts = []
    scale_parts = []
    for i in range(0, N, chunk):
        wc = weight[i : i + chunk].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)
        scale = (maxabs / 7.0).clamp_min(torch.finfo(torch.float16).tiny)
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], K // 8
        )
        parts.append(packed)
        # Keep the compact kernel-side scale representation; it is not a
        # live model parameter and never aliases target tensors.
        scale_parts.append(scale.to(torch.float16))
    qweight_contig = torch.cat(parts, dim=0)
    scales_contig = torch.cat(scale_parts, dim=0)
    qweight = qweight_contig.t()
    scales = scales_contig.t().contiguous()
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(
    model,
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    model_dtype = getattr(head_weight, "dtype", x.dtype)
    # W4A16 is the kernel's private boundary.  Restore BF16 before returning
    # to the draft graph so the next draft/MoE layer cannot receive FP16.
    kernel_x = flat if flat.dtype == torch.float16 else flat.to(torch.float16)
    logits = torch.ops._xpu_C.int4_gemm_w4a16(
        kernel_x, qweight, None, scales, qzeros, group_size, None
    )
    return logits.to(model_dtype).reshape(*x.shape[:-1], qweight.shape[1])


@torch.no_grad()
def build_draft_lmhead_int4(model) -> None:
    if (os.environ.get("B70_DRAFT_LMHEAD_INT4") != "1"
            and os.environ.get("VLLM_XPU_DRAFT_LMHEAD_INT4") != "1"):
        return
    if getattr(model, "_vllm_xpu_lmhead_int4", None) is not None:
        return
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None:
        print("[B70] draft LM head INT4: lm_head.weight unavailable; skip", flush=True)
        return
    print(
        f"[B70] draft LM head INT4: quantizing {tuple(weight.shape)} "
        f"from {weight.dtype} -> INT4 g128 sym (one-time)",
        flush=True,
    )
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(weight.detach())
    model._vllm_xpu_lmhead_int4 = (qweight, scales, qzeros, group_size)
    fp16_bytes = weight.numel() * weight.element_size()
    int4_bytes = qweight.numel() * qweight.element_size() + scales.numel() * scales.element_size()
    print(
        f"[B70] draft LM head INT4: ready. {fp16_bytes/1e9:.2f} GB -> "
        f"{int4_bytes/1e9:.2f} GB INT4; target weights untouched",
        flush=True,
    )


def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    qweight, scales, qzeros, group_size = model._vllm_xpu_lmhead_int4
    logits = int4_lmhead_logits(
        model, hidden_states, qweight, scales, qzeros, group_size
    )
    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)
    if org is not None and logits.shape[-1] > org:
        logits = logits[..., :org]
    return logits
