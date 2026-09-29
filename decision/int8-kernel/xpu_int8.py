# SPDX-License-Identifier: Apache-2.0
# XPU W8A8 INT8 (dynamic per-token symmetric activations, per-channel/per-tensor
# symmetric weights) linear kernel for Intel Xe2 (Arc B580).
#
# Per linear call: 2 launches (cached CompiledKernels launched directly).
#   1. _per_token_quant_int8: one program per row -> int8 row + fp32 scale.
#   2. _i8_gemm_dq: int8 x int8 -> int32 DPAS GEMM (2D block loads) with the
#      dequant epilogue fused: out = acc * x_scale[m] * w_scale[n] (+ bias).
# The int8 weight is stored once as a C-contiguous [K, N] matrix.
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from triton.language.extra import libdevice

from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)


@triton.jit
def _per_token_quant_int8(
    x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm,
    BLOCK: tl.constexpr, COMPAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax / 127.0, 1e-5)
    y = libdevice.rint(x / scale)
    y = tl.minimum(tl.maximum(y, -128.0), 127.0)
    tl.store(q_ptr + row * stride_qm + offs, y.to(tl.int8), mask=mask)
    if COMPAT:
        # Reproduces the Triton INT8 path's quantizer (xpu_ops.dynamic_per_
        # token_int8_quant_ref under torch.compile): it quantizes with the
        # fp32 scale but returns the scale rounded to the activation dtype.
        scale = scale.to(x_ptr.dtype.element_ty).to(tl.float32)
    tl.store(s_ptr + row, scale)


@triton.jit
def _i8_gemm_dq(
    a_ptr, b_ptr, xs_ptr, ws_ptr, bias_ptr, c_ptr, M, N, K,
    stride_am, stride_bk, stride_cm,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_bp = tl.make_block_ptr(
        a_ptr, (M, K), (stride_am, 1), (pid_m * BM, 0), (BM, BK), (1, 0)
    )
    b_bp = tl.make_block_ptr(
        b_ptr, (K, N), (stride_bk, 1), (0, pid_n * BN), (BK, BN), (1, 0)
    )
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_bp = tl.advance(a_bp, (0, BK))
        b_bp = tl.advance(b_bp, (BK, 0))

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    xs = tl.load(xs_ptr + rm, mask=rm < M, other=0.0)
    ws = tl.load(ws_ptr + rn, mask=rn < N, other=0.0)
    y = (acc.to(tl.float32) * xs[:, None] * ws[None, :]).to(c_ptr.dtype.element_ty)
    if HAS_BIAS:
        # added in the output dtype, as triton_scaled_mm does
        y = y + tl.load(bias_ptr + rn, mask=rn < N, other=0.0)[None, :]
    c_bp = tl.make_block_ptr(
        c_ptr, (M, N), (stride_cm, 1), (pid_m * BM, pid_n * BN), (BM, BN), (1, 0)
    )
    tl.store(c_bp, y.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))



_FAST_LAUNCH = os.environ.get("VLLM_XPU_INT8_FAST_LAUNCH", "1") == "1"


def _gemm_config(M: int, N: int):
    # (BM, BN, BK, GROUP_M, num_warps, num_stages, grf_mode), tuned on Arc B580
    # for K in {2560, 4096, 9216}, N in {2560 .. 18432}.
    if M <= 64:
        return (16 if M <= 16 else 32, 256, 64, 4, 8, 3, None)
    if M <= 256:
        if N >= 8192:
            return (64, 256, 64, 4, 16, 3, "256")
        return (256, 128, 64, 4, 32, 3, "256")
    if N >= 8192:
        return (256, 256, 64, 4, 32, 3, "256")
    return (256, 128, 64, 4, 32, 3, "256")


# "compat" (default) = bit-for-bit the Triton INT8 path's activation quantizer,
# so served answers/probabilities (and the fitted calibration) are unchanged.
# "exact" = self-consistent fp32 scale (slightly closer to BF16 per layer).
_QUANT_COMPAT = os.environ.get("VLLM_XPU_INT8_QUANT_MODE", "compat") != "exact"


def _quant_meta(K: int):
    BLOCK = triton.next_power_of_2(K)
    return BLOCK, min(max(BLOCK // 512, 1), 16)


def _launch_quant(x2d, xq, xs):
    M, K = x2d.shape
    BLOCK, nw = _quant_meta(K)
    return _per_token_quant_int8[(M,)](
        x2d, xq, xs, K, x2d.stride(0), xq.stride(0), BLOCK=BLOCK,
        COMPAT=_QUANT_COMPAT, num_warps=nw,
    )


def _launch_gemm(xq, xs, w_kn, ws, bias, out):
    M, K = xq.shape
    N = w_kn.shape[1]
    BM, BN, BK, G, nw, ns, grf = _gemm_config(M, N)
    kw = dict(BM=BM, BN=BN, BK=BK, GROUP_M=G, num_warps=nw, num_stages=ns)
    if grf:
        kw["grf_mode"] = grf
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return _i8_gemm_dq[grid](
        xq, w_kn, xs, ws, bias if bias is not None else ws, out, M, N, K,
        xq.stride(0), w_kn.stride(0), out.stride(0),
        HAS_BIAS=bias is not None, **kw,
    )


# Launch plans: eager prefill is CPU-launch-bound on XPU (Triton's generic
# launcher spends ~20 us/launch in binding + cache-key work; with ~130 int8
# linears x 2 launches that dominated short requests). The first call for a
# given shape/specialization goes through Triton normally; its CompiledKernels
# are cached with everything precomputed, and later calls launch them directly.
_PLANS: dict = {}


def xpu_w8a8_int8_linear(x, w_kn, ws, bias):
    K = x.shape[-1]
    N = w_kn.shape[1]
    x2d = x.reshape(-1, K)
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    M = x2d.shape[0]
    if M == 0:
        return torch.empty((*x.shape[:-1], N), dtype=x.dtype, device=x.device)
    dev = x2d.device
    xq = torch.empty((M, K), dtype=torch.int8, device=dev)
    xs = torch.empty((M,), dtype=torch.float32, device=dev)
    out = torch.empty((M, N), dtype=x.dtype, device=dev)
    has_bias = bias is not None
    b = bias if has_bias else ws
    sxm = x2d.stride(0)
    key = (M, N, K, sxm, x.dtype, has_bias, x2d.data_ptr() & 15 == 0,
           w_kn.data_ptr() & 15 == 0, b.data_ptr() & 15 == 0, w_kn.stride(0))
    plan = _PLANS.get(key) if _FAST_LAUNCH else None
    if plan is None:
        qck = _launch_quant(x2d, xq, xs)
        gck = _launch_gemm(xq, xs, w_kn, ws, bias, out)
        if _FAST_LAUNCH:
            if len(_PLANS) > 65536:
                _PLANS.clear()
            qck._init_handles()
            gck._init_handles()
            BLOCK, _ = _quant_meta(K)
            BM, BN, BK, G = _gemm_config(M, N)[:4]
            _PLANS[key] = (
                qck.run, qck.function, qck.packed_metadata, BLOCK,
                gck.run, gck.function, gck.packed_metadata,
                triton.cdiv(M, BM) * triton.cdiv(N, BN), (has_bias, BM, BN, BK, G),
            )
    else:
        qrun, qfn, qpm, BLOCK, grun, gfn, gpm, g0, cexpr = plan
        stream = torch.xpu.current_stream().sycl_queue
        qrun(M, 1, 1, stream, qfn, qpm, None, None, None,
             x2d, xq, xs, K, sxm, K, BLOCK, _QUANT_COMPAT)
        grun(g0, 1, 1, stream, gfn, gpm, None, None, None,
             xq, w_kn, xs, ws, b, out, M, N, K, K, w_kn.stride(0), N, *cexpr)
    return out.view(*x.shape[:-1], N)


class XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "requires XPU."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.is_static_input_scheme:
            return False, "XPUInt8ScaledMM supports dynamic activation scales only."
        if not c.input_symmetric:
            return False, "XPUInt8ScaledMM supports symmetric activations only."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        w_q = getattr(layer, w_q_name)  # [N, K] int8
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(w_q.data.t().contiguous(), requires_grad=False),
        )  # [K, N] C-contiguous
        weight_scale = getattr(layer, w_s_name)
        if len(layer.logical_widths) > 1 and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        N = w_q.shape[0]
        ws = weight_scale.data.to(torch.float32).reshape(-1)
        if ws.numel() == 1:
            ws = ws.expand(N)
        assert ws.numel() == N, f"weight scale {tuple(ws.shape)} vs N={N}"
        replace_parameter(
            layer, w_s_name, torch.nn.Parameter(ws.contiguous(), requires_grad=False)
        )
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, _, _, _ = self._get_layer_params(layer)
        return xpu_w8a8_int8_linear(x, w_q, w_s, bias)
