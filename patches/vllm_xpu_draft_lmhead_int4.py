"""Optional XPU draft MTP LM-head INT4 g128 symmetric helper.

Provides one-time quantization of the shared draft lm_head and routes the
cuantizacion one-time del lm_head fp16 compartido a GPTQ INT4 g128 sym y el
ruteo de las 4 pasadas del draft por ``int4_gemm_w4a16``. El target queda
fp16 (lossless).
"""
from __future__ import annotations

import os

import torch


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantiza un lm_head fp16 [N, K] a GPTQ INT4 g128 sym.

    Returns (qweight, scales, qzeros, group_size):
      qweight: int32 [K//8, N] en layout NT (strides[-2] == 1), nibbles
               secuenciales LSB-first, valor almacenado = q + 8 (q in [-8, 7])
      scales:  fp16 [K//group_size, N]
      qzeros:  int8 tensor([8])  -> rama simetrica de int4_gemm_w4a16
    """
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
        wc = weight[i : i + chunk].float()  # [c, K] fp32 (chunked: no 5 GB temp)
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)  # [c, g]
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8  # 0..15
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], K // 8
        )
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)  # [N, K//8] int32
    scales_contig = torch.cat(scale_parts, dim=0)  # [N, g] fp16
    # Layout NT requerido por la op (strides[-2] == 1) + scales contiguas
    qweight = qweight_contig.t()  # [K//8, N], strides (1, K//8)
    scales = scales_contig.t().contiguous()  # [g, N]
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Logits [.., vocab] via int4_gemm_w4a16 (mismo formato que el cuerpo)."""
    flat = x.reshape(-1, x.shape[-1])
    logits = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, group_size, None
    )
    return logits.reshape(*x.shape[:-1], qweight.shape[1])


@torch.no_grad()
def build_draft_lmhead_int4(model) -> None:
    """Quantize the draft lm_head once when the opt-in env gate is enabled."""
    if os.environ.get("VLLM_XPU_DRAFT_LMHEAD_INT4") != "1":
        return
    if getattr(model, "_vllm_xpu_lmhead_int4", None) is not None:
        return
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None:
        print("[vllm-xpu] draft LM head INT4: lm_head.weight unavailable; "
              "draft sigue por fp16", flush=True)
        return
    print("[vllm-xpu] draft LM head INT4: quantizing lm_head fp16 "
          f"{tuple(weight.shape)} -> INT4 g128 sym (one-time)", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(
        weight.detach()
    )
    model._vllm_xpu_lmhead_int4 = (qweight, scales, qzeros, group_size)
    fp16_bytes = weight.numel() * weight.element_size()
    int4_bytes = qweight.numel() * qweight.element_size() + (
        scales.numel() * scales.element_size()
    )
    print(f"[vllm-xpu] draft LM head INT4: ready. {fp16_bytes/1e9:.2f} GB fp16 -> "
          f"{int4_bytes/1e9:.2f} GB INT4 (ahorro "
          f"{(fp16_bytes - int4_bytes)/1e6:.1f} MB/lectura)", flush=True)


def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""
    qweight, scales, qzeros, group_size = model._vllm_xpu_lmhead_int4
    logits = int4_lmhead_logits(
        hidden_states, qweight, scales, qzeros, group_size
    )
    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)
    if org is not None and logits.shape[-1] > org:
        logits = logits[..., :org]
    return logits
