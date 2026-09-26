"""Optional XPU draft MTP-layer INT4 g128 symmetric helper.

Provides one-time quantization of the draft MTP linears and routes them
cuantizacion one-time de los 5 linears del modulo MTP del draft a GPTQ
INT4 g128 sym y el ruteo por ``int4_gemm_w4a16``. El target queda intacto.
"""
from __future__ import annotations

import os

import torch


def quantize_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantiza un peso [N, K] a GPTQ INT4 g128 sym (formato int4_gemm_w4a16).

    Returns (qweight, scales, qzeros, group_size):
      qweight: int32 [K//8, N] layout NT (strides[-2] == 1), nibbles
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
        wc = weight[i : i + chunk].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(
            wc.shape[0], K // 8
        )
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)
    scales_contig = torch.cat(scale_parts, dim=0)
    qweight = qweight_contig.t()
    scales = scales_contig.t().contiguous()
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def _collect_linears(predictor) -> list[tuple[str, torch.nn.Module]]:
    """Lista (name, linear) de los 5 linears del MTP predictor a INT4."""
    found: list[tuple[str, torch.nn.Module]] = []
    found.append(("fc", predictor.fc))
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
    """Duck-typed quant_method: apply() rutea por int4_gemm_w4a16."""

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
        if flat.dtype != torch.float16:
            flat = flat.to(torch.float16)
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            flat, self.qweight, None, self.scales, self.qzeros,
            self.group_size, None,
        )
        return out.reshape(*x.shape[:-1], self.qweight.shape[1])


@torch.no_grad()
def build_draft_mtp_int4(model) -> None:
    """Cuantiza los linears del MTP predictor (one-time en load_weights; no-op
    si no hay env gate o si ya se construyo). Almacena en
    model._vllm_xpu_mtp_int4_built."""
    if os.environ.get("VLLM_XPU_DRAFT_MTP_INT4") != "1":
        return
    if getattr(model, "_vllm_xpu_mtp_int4_built", False):
        return
    predictor = getattr(model, "model", None)
    if predictor is None or not hasattr(predictor, "layers"):
        print("[vllm-xpu] draft MTP INT4: no multi-token predictor; "
              "MTP queda en BF16", flush=True)
        return
    linears = _collect_linears(predictor)
    if not linears:
        print("[vllm-xpu] draft MTP INT4: no linears found; skip", flush=True)
        return
    print(f"[vllm-xpu] draft MTP INT4: quantizing {len(linears)} linears "
          f"del MTP -> INT4 g128 sym (one-time)", flush=True)
    total_fp16 = 0
    total_int4 = 0
    for name, lin in linears:
        w = getattr(lin, "weight", None)
        if w is None:
            continue
        orig_shape = tuple(w.shape)
        qweight, scales, qzeros, gs = quantize_to_int4(w.detach())
        lin._vllm_xpu_mtp_int4 = _VllmXpuMTPInt4LinearMethod(
            qweight, scales, qzeros, gs
        )
        lin.quant_method = lin._vllm_xpu_mtp_int4
        fp16_bytes = w.numel() * w.element_size()
        int4_bytes = qweight.numel() * qweight.element_size() + (
            scales.numel() * scales.element_size()
        )
        total_fp16 += fp16_bytes
        total_int4 += int4_bytes
        with torch.no_grad():
            lin.weight.set_(
                torch.empty(0, dtype=w.dtype, device=w.device)
            )
        print(f"[vllm-xpu] draft MTP INT4: {name} {orig_shape} "
              f"{fp16_bytes/1e6:.0f} MB -> {int4_bytes/1e6:.0f} MB "
              f"(fp16 liberado)", flush=True)
    model._vllm_xpu_mtp_int4_built = True
    print(f"[vllm-xpu] draft MTP INT4: ready. {total_fp16/1e9:.2f} GB BF16 -> "
          f"{total_int4/1e9:.2f} GB INT4 (ahorro "
          f"{(total_fp16 - total_int4)/1e6:.0f} MB/lectura)", flush=True)
