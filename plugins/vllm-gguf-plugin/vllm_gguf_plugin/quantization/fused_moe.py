# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, logger


def _xpu_grouped_proj(A, W, qtype, counts, group):
    """A is sorted by expert id (counts[e] rows each). Returns W[e] @ a for every row, via XMX GEMM."""
    import gguf

    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    E, rows, nbytes = W.shape
    cols = nbytes // type_size * block_size
    out = torch.empty((A.shape[0], rows), dtype=A.dtype, device=A.device)
    active = [e for e, c in enumerate(counts) if c]
    compact = len(active) < E // 2
    ids = active if compact else list(range(E))
    start = 0
    for i in range(0, len(ids), group):
        chunk = ids[i:i + group]
        cnt = [counts[e] for e in chunk]
        ntok = sum(cnt)
        if ntok == 0:
            continue
        if compact:
            Wq = W.index_select(0, torch.tensor(chunk, device=W.device, dtype=torch.long))
        else:
            Wq = W[chunk[0]:chunk[-1] + 1]
        g = Wq.shape[0]
        Wd = ops.ggml_dequantize(Wq.reshape(g * rows, nbytes), qtype, g * rows, cols, A.dtype)
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=A[start:start + ntok],
            ptr_A_scale=None,
            # the XPU grouped GEMM takes weights as [experts, in, out]
            ptr_B=Wd.view(g, rows, cols).transpose(1, 2).contiguous(),
            ptr_B_scale=None,
            ptr_bias=None,
            ptr_D=out[start:start + ntok],
            rows_per_expert=torch.tensor(cnt, device=A.device, dtype=torch.int32),
            N=rows,
            K=cols,
            num_experts=g,
        )
        start += ntok
    return out


def _xpu_grouped_moe(x, w1, w2, topk_weights, topk_ids, weight_type, weight_type2, act):
    import os

    import vllm_xpu_kernels._moe_C  # noqa: F401  (registers _moe_C ops)
    import vllm_xpu_kernels._xpu_C  # noqa: F401

    group = int(os.environ.get("VLLM_GGUF_XPU_PREFILL_GROUP", "64"))
    num_tokens, top_k = topk_ids.shape
    E = w1.shape[0]
    hidden = x.shape[1]
    remapped = torch.empty((num_tokens * top_k, hidden), dtype=x.dtype, device=x.device)
    rows_per_expert = torch.zeros((E,), dtype=torch.int32, device=x.device)
    unpermuted = torch.empty((num_tokens, top_k), dtype=torch.int32, device=x.device)
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=x,
        hidden_states_scales=None,
        remapped_hidden_states=remapped,
        remapped_hidden_states_scales=None,
        expert_map=None,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unpermuted,
        topk_ids=topk_ids,
        total_experts_num=E,
        local_experts_num=E,
    )
    counts = rows_per_expert.tolist()
    h = _xpu_grouped_proj(remapped, w1, weight_type, counts, group)
    h = act(h)
    h = _xpu_grouped_proj(h, w2, weight_type2, counts, group)
    out = torch.empty_like(x)
    torch.ops._moe_C.moe_gather(out, h, topk_weights, unpermuted, E)
    return out

def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(activation_enum, out, inp)
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    import os

    out_hidden_states = torch.empty_like(x)
    # On XPU the SYCL kernel takes expert ids directly: no block alignment. Large batches
    # (prefill) dequantize the active experts and use grouped matmuls instead.
    use_sycl = ops.sycl_moe_available(x, weight_type) and ops.sycl_moe_available(x, weight_type2)
    if (
        x.device.type == "xpu"
        and x.shape[0] > int(os.environ.get("VLLM_GGUF_XPU_PREFILL_MIN_TOKENS", "128"))
    ):
        return _xpu_grouped_moe(
            x, w1, w2, topk_weights, topk_ids, weight_type, weight_type2, act
        )
    if (
        not use_sycl
        and weight_type2 in MMQ_QUANT_TYPES
        and weight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(weight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif weight_type2 in MMVQ_QUANT_TYPES and weight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, weight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, weight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = fused_mul_mat_gguf_op(inp, w1[ii], weight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], weight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    del w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13_weight)

        w13_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_weight_type, extra_weight_attrs)
        layer.register_parameter("w13_weight_type", w13_weight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)

        w2_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_weight_type, extra_weight_attrs)
        layer.register_parameter("w2_weight_type", w2_weight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        return fused_moe_gguf_op(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            layer.w13_weight_type.weight_type,
            layer.w2_weight_type.weight_type,
            layer.activation.value,
        )
