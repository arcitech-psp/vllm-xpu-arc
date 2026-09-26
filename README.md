# vLLM XPU Arc

[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Intel Arc XPU](https://img.shields.io/badge/target-Intel%20Arc%20XPU-0071C5.svg)](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc/workstations/a-series.html)
[![Tiel-Coder model](https://img.shields.io/badge/Tiel--Coder-Hugging%20Face-orange.svg)](https://huggingface.co/arcitech-psp/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP)

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/arcitech-logo-white.png"><img src="assets/arcitech-logo-black.png" alt="ArciTech logo"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.png"><img src="assets/hero-light.png" alt="vLLM XPU for Intel Arc"></picture>

This is a reviewable local build recipe for the best working Intel Arc/XPU
route measured with Tiel-Coder: compressed-tensors W4A16 expert weights, the
official BF16 MTP head, and three speculative draft tokens. It is a measured
route, not a certification for every model, driver, or future vLLM release.

## At a glance

- **One Intel Arc Pro B70, 32 GB** served the tested 35B-parameter coding model.
- **116.9 tokens/s for one user; 339.9 tokens/s shared across four** on GPTQ-A.
- **219 of 224** on the internal agentic coding evaluation.
- **Four 131,072-token conversations at once** in the measured configuration.
- **vLLM 0.27.2rc1.dev77+gac7509e2b**, a custom XPU build with PyTorch XPU support.

Tokens per second (tokens/s) is how quickly generated text arrives. A shared
total is the combined output of several concurrent users.

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/test-bench-dark.png"><img src="assets/test-bench-light.png" alt="ArciTech test bench specification card"></picture>

## Test system

The short version: a consumer AM4 desktop with one 32 GB workstation GPU — no
datacenter hardware.

<details>
<summary>Full test system</summary>

| Part | Value |
|---|---|
| GPU 1 (serves the model) | Intel Arc Pro B70, 32 GB |
| GPU 2 (in the machine, not used for these tests) | Intel Arc A310 LP, 4 GB |
| CPU | AMD Ryzen 7 5800X, 8 cores / 16 threads |
| System memory | 32 GB DDR4-3200 (4 × 8 GB) |
| Motherboard | ASUS ROG Strix B550-F Gaming (AM4, PCIe 4.0) |
| Model storage (weights served from here) | 1 TB Samsung PM9A1 NVMe SSD (PCIe 4.0) |
| Other storage (archive only; the model was not loaded from it) | 1.5 TB WD Green HDD |
| OS | Ubuntu 24.04.4 LTS, Linux kernel 7.0 |
| Intel GPU runtime | compute-runtime 26.22.38646.4 (Level Zero + OpenCL), Level Zero loader 1.28.6, IGC 2.11.12 |
| Container | Docker 29.1.3 |
| Serving stack | vLLM 0.27.2rc1.dev77+gac7509e2b (custom XPU build, `vllm-xpu-arc`), PyTorch 2.13.0+xpu, vllm-xpu-kernels 0.1.12.3 |
| Serving settings | FP8 KV cache, 131,072-token context, 4 concurrent sequences, 4,096 max batched tokens, MTP with 3 draft tokens |

</details>

## Read the measured results

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/speed-comparison-dark.png"><img src="assets/speed-comparison-light.png" alt="Per-user and shared total decode speed at one, two, and four users"></picture>

| Build | 1 user | 2 users | 4 users |
|---|---:|---:|---:|
| GPTQ-A, per-stream / aggregate tok/s | 116.9 / 113.4 | 105.9, 113.6 / 195.6 | 94.0, 92.5, 92.5, 93.0 / 339.9 |
| FAST CT2, per-stream / aggregate tok/s | 132.0 / 127.7 | 120.8, 125.0 / 228.6 | 99.3, 101.1, 101.0, 100.4 / 374.8 |

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/quality-comparison-dark.png"><img src="assets/quality-comparison-light.png" alt="Internal 224-task coding evaluation comparison"></picture>

| Build | Total | Code (184) | Tool (20) | Edit (20) |
|---|---:|---:|---:|---:|
| GPTQ-A | 219 / 224 | 181 | 18 | 20 |
| FAST CT2 | 213–215 / 224 | 176–178 | 18 | 19 |

This internal evaluation is aggregate-only; task prompts and per-task records
are not released.

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/precision-split-dark.png"><img src="assets/precision-split-light.png" alt="Logical precision split"></picture>

The logical parameter split is approximately 93.5% routed expert GPTQ int4 and
6.5% BF16. Attention, routing, normalization, embeddings, shared experts, the
vision tower, output head, and MTP tensors remain BF16.

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/mtp-acceptance-dark.png"><img src="assets/mtp-acceptance-light.png" alt="MTP acceptance by draft position"></picture>

MTP proposes tokens ahead and lets the main model verify several at once when
the guesses match. GPTQ-A acceptance by configured draft position was 74.3%,
50.9%, and 35.6%.

## How to read this

- **Token:** a small piece of text; words may be one or several tokens.
- **Tokens/s:** generated tokens per second, a practical speed measure.
- **MoE:** mixture of experts; only selected experts handle each token.
- **int4 / BF16:** compact 4-bit routed-expert weights and retained 16-bit tensors.
- **MTP:** multi-token prediction; a draft path proposes tokens for verification.
- **KV cache:** saved attention state that avoids recomputing the conversation so far.
- **Context:** the maximum conversation length the model can consider at once.

## Best working path

The default route uses the compressed-tensors Tiel-Coder deployment and was
measured with vLLM `0.27.2rc1.dev77+gac7509e2b`, PyTorch `2.13.0+xpu`,
`vllm-xpu-kernels 0.1.12.3`, FP8 KV cache, a 131,072-token maximum context,
four sequence slots, 4,096 maximum batched tokens, and three MTP drafts.

The repository contains source and build recipes, not model weights or runtime
caches. Build the pinned core patch and image, then mount the model read-only
using the launcher supplied with the release. Set the launcher's model input to
the model you downloaded from the
[Hugging Face model page](https://huggingface.co/arcitech-psp/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP).

```bash
git clone https://github.com/vllm-project/vllm.git
git -C vllm checkout ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9
./scripts/apply-vllm-core-patch.sh vllm
docker build -t vllm-xpu-arc:local .
```

Use the model's Tiel Sharp template, BF16 compute, FP8 KV cache, four 131K
slots, three MTP drafts, and the parser settings shown in the model card.
Re-measure capacity for a different model, driver, or slot count.

## For practitioners

### Feature matrix

Status labels describe the measured boundary. `working` means the path was
exercised in the campaign; `partial` and `experimental` are included for
review and future work, not as the default launch route.

| Feature | Status | Tested boundary |
|---|---|---|
| Compressed-tensors W4A16 fused int4 MoE | working | Tiel-Coder body, group 128, native XPU fused-MoE path. |
| MTP speculative decoding | working | Official BF16 MTP head, three drafts, measured acceptance reported above. |
| FP8 KV cache | working | Used in the measured Tiel-Coder and long-context runs. |
| Tool calling | working | `qwen3_coder`, including streamed calls. |
| Reasoning parser | working | `qwen3` reasoning/content split on the tested path. |
| Vision input | working | Exercised with a 4 MP processing limit. |
| 4 × 128K concurrency | working | Four 131,072-token slots fit the measured KV pool. |
| MXFP4 W4A16 | partial | oneDNN/SYCL source and build gate included; not the default path. |
| XPU graphs | partial | Exact-length C1 dispatch is gated and hardware-sensitive. |
| GGUF k-quant MoE | experimental | Q3_K/Q4_K/Q5_K/Q6_K SYCL source is included in the plugin. |
| Adaptive MTP | experimental | Opt-in controller for 2–8 drafts; no verifier or sampling change. |
| Gated DeltaNet snapshot copy | experimental | Opt-in XPU state-copy contract. |
| DFlash2 / DSpark | UPSTREAM | Use v0.30.0's upstream implementations; the local adaptive MTP path is not a substitute. |
| Weight and KV offload tiers | UPSTREAM | Present in v0.30.0; not implemented by this repository's patch. |

### Experimental v0.30.0 rebase

The rebase under `experimental/v0.30-rebase/` applies cleanly at source level,
but **has not been built or served**. Stock v0.30.0 loaded GPTQ-A and reached
compile/warmup, then segfaulted in stock SYCL top-k; a no-graph retry failed the
XPU memory reservation before a model endpoint became available. Stock
performance was not measured. Use the default build for the measured route.

The rebase keeps upstream v0.30 features separate from local XPU deltas:
mixed XPU GDN dispatch, the XPU GDN snapshot-copy contract, adaptive MTP and
exact C1 graph gates, optional oneDNN MXFP4 W4A16 and draft int4, and the GGUF
XPU plugin/SYCL k-quant MoE path. Those optional paths are not the release
benchmark.

### Included source

- `patches/0001-vllm-ac7509e2b-xpu-extras.patch` — measured XPU core delta.
- `patches/0002-vllm-gguf-plugin-56bfc18.patch` — included GGUF plugin delta.
- `experimental/v0.30-rebase/` — source-level rebase, not built or served.
- `plugins/vllm-gguf-plugin/` — plugin source and tests.
- `adaptive/` — optional adaptive MTP and Gated DeltaNet SYCL sources.
- `mxfp4/` — optional oneDNN-backed MXFP4 W4A16 sources.
- `scripts/` — patch and container-run helpers.

## Limits and privacy

- Working measurements were made on one Intel Arc Pro B70.
- Other Intel GPUs, CUDA, stock-vLLM equivalence, and future driver combinations are untested.
- The measured boundary is four 131,072-token slots; larger contexts require a new capacity measurement.
- The Docker build was not executed on this preparation machine; source, patch, compile, and privacy checks were performed.
- No calibration data or private prompts are included.

## Credits

This work stands on the vLLM project and its Intel XPU contributors, Intel's
Arc hardware and XPU software stack, the Ornith team, `peculiar-ragdoll` for
Tiel and the Sharp template, biMEMO's earlier reference work, and the Hugging
Face community. The related model is published at the
[Hugging Face account `arcitech-psp`](https://huggingface.co/arcitech-psp).

## Feedback and contact

Feedback form: [Google Form](https://docs.google.com/forms/d/1gaUBeulGlZwo8gt4eucGpg3biCKy-tli79urdTesXSI/viewform).
Direct contact: [parthpatel266@gmail.com](mailto:parthpatel266@gmail.com).
GitHub account: [arcitech-psp](https://github.com/arcitech-psp).

## License and related pages

Upstream vLLM and plugin files retain their Apache-2.0 notices. New code in
this repository follows Apache-2.0 as documented in `LICENSE` and `NOTICE`.

- [Tiel-Coder data repository](https://github.com/arcitech-psp/tiel-coder-xpu)
- [Tiel-Coder model card](https://huggingface.co/arcitech-psp/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP)
- [How Tiel-Coder XPU was built](../docs/APPROACH.md)
