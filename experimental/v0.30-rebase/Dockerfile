# Reproducible public-base build for vLLM XPU extensions.
# The pinned base is the public vLLM XPU v0.30.0 image pulled on
# 2026-09-25. The local patch is rebased onto the matching v0.30.0 source.
FROM vllm/vllm-openai-xpu@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends patch \
 && rm -rf /var/lib/apt/lists/*

COPY experimental/v0.30-rebase/patches/0001-vllm-v0.30.0-ced6857-xpu-extras.patch /opt/vllm-xpu-arc/patches/
COPY patches/vllm_xpu_draft_lmhead_int4.py /opt/vllm-xpu-arc/patches/
COPY patches/vllm_xpu_draft_mtp_int4.py /opt/vllm-xpu-arc/patches/

# The public XPU image installs vLLM into the image virtualenv.  This is the
# runtime form of the same source patch; the patch file remains the reviewable
# artifact against the exact upstream checkout.
RUN VLLM_SITE=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
 && patch --batch --forward -p1 -d "$VLLM_SITE" \
      < /opt/vllm-xpu-arc/patches/0001-vllm-v0.30.0-ced6857-xpu-extras.patch \
 && install -m 0644 /opt/vllm-xpu-arc/patches/vllm_xpu_draft_lmhead_int4.py \
      "$VLLM_SITE/vllm/model_executor/models/vllm_xpu_draft_lmhead_int4.py" \
 && install -m 0644 /opt/vllm-xpu-arc/patches/vllm_xpu_draft_mtp_int4.py \
      "$VLLM_SITE/vllm/model_executor/models/vllm_xpu_draft_mtp_int4.py"

RUN /opt/venv/bin/pip install --no-cache-dir 'gguf>=0.17.0'
COPY plugins/vllm-gguf-plugin /opt/vllm-xpu-arc/vllm-gguf-plugin
RUN cd /opt/vllm-xpu-arc/vllm-gguf-plugin \
 && VLLM_GGUF_PLUGIN_SKIP_EXT=1 VLLM_GGUF_BUILD_CUDA=0 \
      /opt/venv/bin/pip install --no-cache-dir --no-deps --no-build-isolation -e . \
 && /opt/venv/bin/python vllm_gguf_plugin/xpu_kernels/build_gguf_xpu.py \
 && test -f vllm_gguf_plugin/xpu_kernels/gguf_xpu_moe.so

COPY adaptive /opt/vllm-xpu-arc/adaptive
COPY mxfp4 /opt/vllm-xpu-arc/mxfp4

# Build the optional SYCL extensions during image construction.  These scripts
# assert that torch.xpu is not initialized while compiling.
RUN /opt/venv/bin/python /opt/vllm-xpu-arc/adaptive/build_kernel.py \
 && /opt/venv/bin/python /opt/vllm-xpu-arc/mxfp4/build_mxfp4_w4a16.py --onednn installed

ENV PYTHONPATH=/opt/vllm-xpu-arc/adaptive:/opt/vllm-xpu-arc/mxfp4 \
    VLLM_XPU_ADAPTIVE_GDN_LIBRARY=/opt/vllm-xpu-arc/adaptive/kernel-build/vllm_xpu_adaptive_gdn.so \
    VLLM_XPU_MXFP4_W4A16_LIBRARY=/opt/vllm-xpu-arc/mxfp4/mxfp4-w4a16-build/vllm_xpu_mxfp4_woq.so \
    VLLM_XPU_ADAPTIVE_MTP=0 \
    VLLM_XPU_ADAPTIVE_GRAPHS=0 \
    VLLM_XPU_MXFP4_W4A16=0 \
    VLLM_XPU_MTP_POLICY=none \
    VLLM_XPU_MTP_BF16_DRAFT=0 \
    VLLM_XPU_DRAFT_LMHEAD_INT4=0 \
    VLLM_XPU_DRAFT_MTP_INT4=0 \
    VLLM_GGUF_USE_CUDA=0 \
    VLLM_GGUF_DENSE_FORMAT=fp8 \
    VLLM_CACHE_ROOT=/cache/vllm-fp8dense \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton \
    SYCL_CACHE_PERSISTENT=1
