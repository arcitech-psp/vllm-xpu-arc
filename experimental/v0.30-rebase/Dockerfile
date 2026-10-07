# Own source patches on official vLLM; CPU-only Intel oneAPI build stage.
FROM intel/oneapi@sha256:5a71507a519c7c05302b53fa32487304df2ee211d6d737d7d948eb53726d4d6f AS oneapi
FROM vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d AS build
USER root
ARG MAX_JOBS=2
ENV MAX_JOBS=${MAX_JOBS}
COPY --from=oneapi /opt/intel/oneapi/compiler/2026.0 /opt/intel/oneapi/compiler/2026.0
COPY --from=oneapi /opt/intel/oneapi/dnnl/2026.0 /opt/intel/oneapi/dnnl/2026.0
COPY experimental/v0.30-rebase/patches /opt/vllm-xpu-arc/source/experimental/v0.30-rebase/patches
COPY patches/vllm_xpu_draft_lmhead_int4.py patches/vllm_xpu_draft_mtp_int4.py /opt/vllm-xpu-arc/source/patches/
COPY plugins/vllm-gguf-plugin /opt/vllm-xpu-arc/source/plugins/vllm-gguf-plugin
COPY adaptive /opt/vllm-xpu-arc/source/adaptive
COPY mxfp4 /opt/vllm-xpu-arc/source/mxfp4
COPY scripts/install-v030.py /opt/vllm-xpu-arc/source/scripts/install-v030.py
RUN /opt/venv/bin/python /opt/vllm-xpu-arc/source/scripts/install-v030.py

# Keep the compiler in the retained build stage. The serving image only receives
# our runtime source, compiled operators and the oneDNN runtime they link against.
FROM vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d
COPY --from=build /opt/vllm-xpu-arc/runtime-site /opt/venv/lib/python3.12/site-packages
COPY --from=build /opt/vllm-xpu-arc/source/adaptive /opt/vllm-xpu-arc/adaptive
COPY --from=build /opt/vllm-xpu-arc/source/mxfp4 /opt/vllm-xpu-arc/mxfp4
COPY --from=build /opt/intel/oneapi/dnnl/2026.0 /opt/intel/oneapi/dnnl/2026.0
COPY scripts/holo4-entrypoint.py /opt/vllm-xpu-arc/scripts/holo4-entrypoint.py
COPY experimental/v0.30-rebase/patches /opt/vllm-xpu-arc/patches
ENV PYTHONPATH=/opt/vllm-xpu-arc/adaptive:/opt/vllm-xpu-arc/mxfp4 \
    VLLM_XPU_ADAPTIVE_GDN_LIBRARY=/opt/vllm-xpu-arc/adaptive/kernel-build/vllm_xpu_adaptive_gdn.so \
    VLLM_XPU_MXFP4_W4A16_LIBRARY=/opt/vllm-xpu-arc/mxfp4/mxfp4-w4a16-build/vllm_xpu_mxfp4_woq.so \
    VLLM_XPU_ADAPTIVE_MTP=0 VLLM_XPU_ADAPTIVE_GRAPHS=0 \
    VLLM_XPU_MXFP4_W4A16=0 VLLM_XPU_MTP_POLICY=none \
    VLLM_XPU_MTP_BF16_DRAFT=0 VLLM_XPU_DRAFT_LMHEAD_INT4=0 \
    VLLM_XPU_DRAFT_MTP_INT4=0 VLLM_GGUF_USE_CUDA=0 VLLM_GGUF_DENSE_FORMAT=fp8 \
    VLLM_CACHE_ROOT=/cache/vllm-holo4 TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton SYCL_CACHE_PERSISTENT=0 \
    VLLM_XPU_USE_SAMPLER_KERNEL=0 VLLM_XPU_INC_WNA16_BACKEND=auto \
    VLLM_XPU_SAFE_BF16_TOPK=1
