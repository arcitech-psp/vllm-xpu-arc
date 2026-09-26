# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from .layout import GGUFLinearLayout
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


def _dense_dequant_enabled() -> bool:
    """VLLM_GGUF_DEQUANT_DENSE: 1 on, 0 off, auto (default) on for XPU."""
    import os

    mode = os.environ.get("VLLM_GGUF_DEQUANT_DENSE", "auto").lower()
    if mode in ("1", "true", "on"):
        return True
    if mode in ("0", "false", "off"):
        return False
    from vllm.platforms import current_platform

    return current_platform.is_xpu()


def _fp8_dense_wanted(layer: torch.nn.Module) -> bool:
    import os

    if os.environ.get("VLLM_GGUF_DENSE_FORMAT", "bf16").lower() != "fp8":
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_xpu():
        return False
    if type(layer).__name__ == "ParallelLMHead":
        return os.environ.get("VLLM_GGUF_FP8_LM_HEAD", "0") == "1"
    return True


def _convert_dense_to_fp8(layer: torch.nn.Module) -> None:
    import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers fp8_gemm_w8a16)

    weight = layer.weight.data.float()  # [out, in], exact dequantized values
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale = (weight.abs().amax(dim=1, keepdim=True) / fmax).clamp_min(1e-12)
    q = (weight / scale).clamp_(-fmax, fmax).to(torch.float8_e4m3fn)
    del weight
    # fp8_gemm_w8a16 takes the weight as an [in, out] view and a [1, out] scale.
    layer.weight.data = q
    layer.weight.data = layer.weight.data.t()
    layer.register_parameter(
        "gguf_fp8_scale",
        torch.nn.Parameter(scale.t().contiguous().to(torch.float32), requires_grad=False),
    )
    layer.gguf_fp8_dense = True


def _fused_mul_mat_gguf(
    x: torch.Tensor, weight: torch.Tensor, weight_type: int
) -> torch.Tensor:
    if weight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if weight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if weight.shape[0] > 5120 else 6
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    if weight_type in UNQUANTIZED_TYPES:
        return x @ weight.T
    if x.shape[0] <= mmvq_safe and weight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(weight, x, weight_type, weight.shape[0])
    elif weight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(weight, x, weight_type, weight.shape[0])
    elif weight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[weight_type]
        shape = (weight.shape[0], weight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(weight, weight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        weight_type = WeightType(weight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {weight_type}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    def __init__(
        self,
        quant_config,
        layout: GGUFLinearLayout | None = None,
    ) -> None:
        self.quant_config = quant_config
        self.layout = layout

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            weight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            weight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
            },
        )
        set_weight_attrs(weight_type, extra_weight_attrs)
        layer.register_parameter("weight_type", weight_type)

        if self.layout is not None:
            set_weight_attrs(
                weight,
                {
                    "gguf_layout": self.layout,
                    "gguf_logical_input_size": input_size,
                    "gguf_weight_type_parameter": weight_type,
                },
            )

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        weight_type = layer.weight_type.weight_type
        if not (weight_type in UNQUANTIZED_TYPES or weight_type in DEQUANT_TYPES):
            weight_type = WeightType(weight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {weight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)
        if _dense_dequant_enabled() and self._dense_dequant_applies(layer):
            self._dequantize_dense_after_load(layer)

    def _dense_dequant_applies(self, layer: torch.nn.Module) -> bool:
        # The input embedding is a lookup: keep it quantized, rows are dequantized on demand.
        return type(layer).__name__ != "VocabParallelEmbedding"

    def _dequantize_dense_after_load(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        weight_type = layer.weight_type
        dtype = getattr(self, "params_dtype", None) or torch.bfloat16
        shard_ids = list(weight.shard_id) if weight.shard_id else []

        def dequant(data: torch.Tensor, qtype: int) -> torch.Tensor:
            if qtype in UNQUANTIZED_TYPES:
                return data.to(dtype)
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            rows = data.shape[0]
            cols = data.shape[1] // type_size * block_size
            out_dtype = torch.float32 if _fp8_dense_wanted(layer) else dtype
            parts = [
                ops.ggml_dequantize(data[i:i + 16384].contiguous(), qtype,
                                    min(16384, rows - i), cols, out_dtype)
                for i in range(0, rows, 16384)
            ]
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        if shard_ids and getattr(weight, "shard_offset_map", None):
            parts = []
            for idx in shard_ids:
                start, end, size = weight.shard_offset_map[idx]
                qtype = weight_type.shard_weight_type.get(idx, weight_type.weight_type)
                parts.append(dequant(weight.data[start:end, :size], qtype))
            dense = torch.cat(parts, dim=0)
        else:
            qtype = weight_type.weight_type
            if qtype in UNQUANTIZED_TYPES:
                return
            dense = dequant(weight.data, qtype)

        new_param = GGUFWeightParameter(
            data=dense,
            weight_loader=weight.weight_loader,
            input_dim=weight.input_dim,
            output_dim=weight.output_dim,
            tensor_shape=weight.tensor_shape,
        )
        new_param.data_container = []
        new_param.shard_id = shard_ids
        new_param.shard_id_map = dict(getattr(weight, "shard_id_map", {}))
        if hasattr(weight, "ignore_warning"):
            new_param.ignore_warning = weight.ignore_warning
        if getattr(weight, "shard_offset_map", None):
            set_weight_attrs(new_param, {"shard_offset_map": dict(weight.shard_offset_map)})
        layer.register_parameter("weight", new_param)
        if _fp8_dense_wanted(layer):
            _convert_dense_to_fp8(layer)
        dense_type = int(WeightType.BF16 if dtype == torch.bfloat16 else WeightType.F16)
        weight_type.weight_type = dense_type
        for idx in list(weight_type.shard_weight_type):
            weight_type.shard_weight_type[idx] = dense_type

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_weight(layer)
        self._materialize_weight_type(layer)

    def _materialize_weight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "weight")

    def _materialize_weight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "weight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        weight = layer.weight
        shard_id_map = weight.shard_id_map
        shard_id = weight.shard_id
        if len(data_container := weight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=weight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=weight.weight_loader,
                input_dim=weight.input_dim,
                output_dim=weight.output_dim,
                tensor_shape=weight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(weight.shard_id_map)
            if hasattr(weight, "ignore_warning"):
                padded_param.ignore_warning = weight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            weight.data_container.clear()
            weight.shard_id.clear()
            weight.shard_id_map.clear()
            if weight.data.numel() > 0:
                weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
            layer.register_parameter("weight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        if self.layout is not None:
            x = self.layout.input_to_gguf(x)

        if getattr(layer, "gguf_fp8_dense", False):
            shape = x.shape
            y = torch.ops._xpu_C.fp8_gemm_w8a16(
                x.reshape(-1, shape[-1]), layer.weight, layer.gguf_fp8_scale, bias
            )
            return y.view(*shape[:-1], y.shape[-1])

        shard_id = layer.weight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            weight = layer.weight
            fallback_wtype = layer.weight_type.weight_type
            shard_weight_types = [
                layer.weight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf_op(x, weight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.weight.shard_offset_map[idx]
                weight_type = layer.weight_type.shard_weight_type.get(
                    idx, fallback_wtype
                )
                result.append(
                    fused_mul_mat_gguf_op(
                        x, weight[start:end, :offset].contiguous(), weight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            weight = layer.weight
            weight_type = layer.weight_type.weight_type
            out = fused_mul_mat_gguf_op(x, weight, weight_type)
        if bias is not None:
            out.add_(bias)
        return out
