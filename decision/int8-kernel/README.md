# Fused INT8 W8A8 linear kernel for Intel Arc (vLLM XPU)

For compressed-tensors INT8 W8A8 checkpoints, vLLM picks its generic `TritonInt8ScaledMMLinearKernel`. On Arc in eager mode, that path is limited by kernel launches, not by math. Each linear runs several small kernels, so a decision request costs a flat ~220 ms even when it is short.

`xpu_int8.py` adds `XPUInt8ScaledMMLinearKernel`, which needs 2 launches per linear call:
1. Per-token symmetric INT8 quantization of the activations (one program per row).
2. An INT8 × INT8 → INT32 DPAS GEMM with the dequant fused into it: `out = acc * x_scale[m] * w_scale[n] (+ bias)`. The weight is stored once as a contiguous [K, N] matrix.

Compiled kernels are cached and launched directly, which halves CPU cost per call.

`linear_init_patched.py` is vLLM's `model_executor/kernels/linear/__init__.py` from the image this repo builds (vLLM 0.27.2rc1). The only change: it registers the new kernel ahead of Triton for XPU. `../serve-decision.sh` mounts both files automatically for INT8 checkpoints (`INT8_KERNEL=auto`).

**Correctness:** outputs are bit-identical to the Triton path, from M = 1 to 4000, including odd shapes and bias.

**Speed** on an Arc B580, Mica-v0.1-4B INT8, median of 20 warm requests:

| Request | This kernel | vLLM Triton INT8 | FP8 weights (for comparison) |
|---|---|---|---|
| short (168 tokens) | 52 ms | 213 ms | 50 ms |
| ~1K tokens | 181 ms | 1,001 ms | 188 ms |
| ~4K tokens | 600 ms | 3,665 ms | 679 ms |

**Known limit:** short requests are bound by CPU launch overhead. Fusing the quantization into the preceding RMSNorm would remove one launch per linear.
