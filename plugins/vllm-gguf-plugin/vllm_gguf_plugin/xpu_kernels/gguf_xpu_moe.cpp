// SYCL kernel for GGUF k-quant MoE projections on Intel XPU (vLLM XPU + vllm-gguf-plugin).
//
// gguf_xpu::moe_vec(x, w, ids, top_k, qtype, rows) -> y
//   x   [N, in]            float input rows
//   w   [E, rows, bytes]   uint8 GGUF blocks, one row of blocks per output row
//   ids [N * top_k] or [T, K] expert per (token, slot), flattened row-major
//   y   [N * top_k, rows]  y[p] = W[ids[p]] @ x[p / top_k]
// Each output element dequantizes one row of blocks and takes its dot product with the input in
// float32, so the result equals an exact ggml dequantization followed by a matmul.
// Block layouts and per-value formulas follow ggml (QK_K = 256).
#include <limits>

#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

namespace {

constexpr int QK_K = 256;
constexpr int64_t TYPE_Q3_K = 11, TYPE_Q4_K = 12, TYPE_Q5_K = 13, TYPE_Q6_K = 14;

inline float fp16_to_float(uint8_t lo, uint8_t hi) {
  uint16_t h = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
  int sign = (h >> 15) & 1;
  int exp = (h >> 10) & 0x1F;
  int mant = h & 0x3FF;
  float v;
  if (exp == 0) {
    v = sycl::ldexp(static_cast<float>(mant), -24);           // subnormal: mant * 2^-24
  } else if (exp == 31) {
    v = mant ? sycl::nan(0u) : std::numeric_limits<float>::infinity();
  } else {
    v = sycl::ldexp(static_cast<float>(mant | 0x400), exp - 25);  // (1024 + mant) * 2^(exp-25)
  }
  return sign ? -v : v;
}

inline void scale_min_k4(int j, const uint8_t *q, int &d, int &m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Q4_K: d(2) dmin(2) scales(12) qs(128) = 144 bytes
inline float dot_q4_k(const uint8_t *blk, const float *x) {
  const float d = fp16_to_float(blk[0], blk[1]);
  const float dmin = fp16_to_float(blk[2], blk[3]);
  const uint8_t *scales = blk + 4;
  const uint8_t *qs = blk + 16;
  float acc = 0.0f;
  for (int il = 0; il < 4; ++il) {
    int sc, m;
    scale_min_k4(2 * il, scales, sc, m);
    const float d1 = d * sc, m1 = dmin * m;
    scale_min_k4(2 * il + 1, scales, sc, m);
    const float d2 = d * sc, m2 = dmin * m;
    const uint8_t *q = qs + 32 * il;
    const float *xa = x + 64 * il;
    const float *xb = xa + 32;
    float s1 = 0.0f, t1 = 0.0f, s2 = 0.0f, t2 = 0.0f;
    for (int l = 0; l < 32; ++l) {
      s1 += xa[l] * static_cast<float>(q[l] & 0xF);
      t1 += xa[l];
      s2 += xb[l] * static_cast<float>(q[l] >> 4);
      t2 += xb[l];
    }
    acc += d1 * s1 - m1 * t1 + d2 * s2 - m2 * t2;
  }
  return acc;
}

// Q5_K: d(2) dmin(2) scales(12) qh(32) qs(128) = 176 bytes
inline float dot_q5_k(const uint8_t *blk, const float *x) {
  const float d = fp16_to_float(blk[0], blk[1]);
  const float dmin = fp16_to_float(blk[2], blk[3]);
  const uint8_t *scales = blk + 4;
  const uint8_t *qh = blk + 16;
  const uint8_t *qs = blk + 48;
  float acc = 0.0f;
  for (int il = 0; il < 4; ++il) {
    int sc, m;
    scale_min_k4(2 * il, scales, sc, m);
    const float d1 = d * sc, m1 = dmin * m;
    scale_min_k4(2 * il + 1, scales, sc, m);
    const float d2 = d * sc, m2 = dmin * m;
    const uint8_t u1 = static_cast<uint8_t>(1 << (2 * il));
    const uint8_t u2 = static_cast<uint8_t>(2 << (2 * il));
    const uint8_t *q = qs + 32 * il;
    const float *xa = x + 64 * il;
    const float *xb = xa + 32;
    float s1 = 0.0f, t1 = 0.0f, s2 = 0.0f, t2 = 0.0f;
    for (int l = 0; l < 32; ++l) {
      const int v1 = (q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
      const int v2 = (q[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
      s1 += xa[l] * static_cast<float>(v1);
      t1 += xa[l];
      s2 += xb[l] * static_cast<float>(v2);
      t2 += xb[l];
    }
    acc += d1 * s1 - m1 * t1 + d2 * s2 - m2 * t2;
  }
  return acc;
}

// Q6_K: ql(128) qh(64) scales(16, int8) d(2) = 210 bytes
inline float dot_q6_k(const uint8_t *blk, const float *x) {
  const uint8_t *ql = blk;
  const uint8_t *qh = blk + 128;
  const int8_t *sc = static_cast<const int8_t *>(static_cast<const void *>(blk + 192));
  const float d = fp16_to_float(blk[208], blk[209]);
  float acc = 0.0f;
  // element e = 128*ip + 32*h + il; qh byte qh[32*ip + il]; scale sc[8*ip + il/16 + 2*h]
  for (int ip = 0; ip < 2; ++ip) {
    for (int h = 0; h < 4; ++h) {
      const float *xs = x + 128 * ip + 32 * h;
      const uint8_t *qb = ql + 64 * ip + 32 * (h & 1);
      const uint8_t *hb = qh + 32 * ip;
      const int shift = 2 * h;
      for (int half = 0; half < 2; ++half) {
        const float scale = d * sc[8 * ip + half + 2 * h];
        float s = 0.0f;
        for (int il = 16 * half; il < 16 * half + 16; ++il) {
          const int low = (h < 2) ? (qb[il] & 0xF) : (qb[il] >> 4);
          const int hi = (hb[il] >> shift) & 3;
          s += xs[il] * static_cast<float>((low | (hi << 4)) - 32);
        }
        acc += scale * s;
      }
    }
  }
  return acc;
}

// Q3_K: hmask(32) qs(64) scales(12) d(2) = 110 bytes
inline int q3_k_scale(int is, const uint8_t *s) {
  if (is < 4) return (s[is] & 0xF) | (((s[is + 8] >> 0) & 3) << 4);
  if (is < 8) return (s[is] & 0xF) | (((s[is + 4] >> 2) & 3) << 4);
  if (is < 12) return (s[is - 8] >> 4) | (((s[is] >> 4) & 3) << 4);
  return (s[is - 8] >> 4) | (((s[is - 4] >> 6) & 3) << 4);
}

inline float dot_q3_k(const uint8_t *blk, const float *x) {
  const uint8_t *hm = blk;
  const uint8_t *qs = blk + 32;
  const uint8_t *scales = blk + 96;
  const float d = fp16_to_float(blk[108], blk[109]);
  float acc = 0.0f;
  // element e = 128*n + 32*j + l; is = 8n + 2j + l/16; bit m = 1 << (4n + j); shift 2j
  for (int n = 0; n < 2; ++n) {
    const uint8_t *q = qs + 32 * n;
    for (int j = 0; j < 4; ++j) {
      const float *xs = x + 128 * n + 32 * j;
      const uint8_t mbit = static_cast<uint8_t>(1 << (4 * n + j));
      const int shift = 2 * j;
      for (int half = 0; half < 2; ++half) {
        const float dl = d * static_cast<float>(q3_k_scale(8 * n + 2 * j + half, scales) - 32);
        float s = 0.0f;
        for (int l = 16 * half; l < 16 * half + 16; ++l) {
          const int v = ((q[l] >> shift) & 3) - ((hm[l] & mbit) ? 0 : 4);
          s += xs[l] * static_cast<float>(v);
        }
        acc += dl * s;
      }
    }
  }
  return acc;
}

int64_t block_bytes(int64_t qtype) {
  switch (qtype) {
    case TYPE_Q3_K: return 110;
    case TYPE_Q4_K: return 144;
    case TYPE_Q5_K: return 176;
    case TYPE_Q6_K: return 210;
    default: return 0;
  }
}

template <typename DotFn>
void run_moe(sycl::queue &q, const float *x, const uint8_t *w, const int32_t *ids, float *y,
             int64_t pairs, int64_t rows, int64_t top_k, int64_t in_features, int64_t nblk,
             int64_t bsize, int64_t stride_e, int64_t stride_r, DotFn dot) {
  q.parallel_for(sycl::range<2>(pairs, rows), [=](sycl::item<2> it) {
     const int64_t p = it[0];
     const int64_t r = it[1];
     const int64_t e = ids[p];
     const float *xr = x + (p / top_k) * in_features;
     const uint8_t *row = w + e * stride_e + r * stride_r;
     float acc = 0.0f;
     for (int64_t b = 0; b < nblk; ++b) acc += dot(row + b * bsize, xr + b * QK_K);
     y[p * rows + r] = acc;
   });
}

torch::Tensor moe_vec(torch::Tensor x, torch::Tensor w, torch::Tensor ids, int64_t top_k,
                      int64_t qtype, int64_t rows) {
  TORCH_CHECK(x.device().is_xpu() && w.device().is_xpu() && ids.device().is_xpu(), "XPU tensors required");
  TORCH_CHECK(w.dtype() == torch::kUInt8 && w.dim() == 3, "w must be uint8 [E, rows, bytes]");
  const int64_t bsize = block_bytes(qtype);
  TORCH_CHECK(bsize > 0, "unsupported GGUF type ", qtype);
  TORCH_CHECK(w.size(1) == rows && w.size(2) % bsize == 0, "bad weight shape");
  const int64_t nblk = w.size(2) / bsize;
  const int64_t in_features = nblk * QK_K;
  TORCH_CHECK(x.dim() == 2 && x.size(1) == in_features, "x must be [N, ", in_features, "]");
  const auto out_dtype = x.dtype();
  auto xf = x.to(torch::kFloat32).contiguous();
  auto wc = w.contiguous();
  auto idf = ids.to(torch::kInt32).reshape({-1}).contiguous();
  const int64_t pairs = x.size(0) * top_k;
  TORCH_CHECK(idf.numel() == pairs, "ids has ", idf.numel(), " entries, expected ", pairs);
  auto y = torch::empty({pairs, rows}, xf.options());
  if (pairs == 0) return y.to(out_dtype);
  auto &queue = c10::xpu::getCurrentXPUStream(x.device().index()).queue();
  const float *xp = xf.data_ptr<float>();
  const uint8_t *wp = wc.data_ptr<uint8_t>();
  const int32_t *ip = idf.data_ptr<int32_t>();
  float *yp = y.data_ptr<float>();
  const int64_t stride_e = wc.stride(0), stride_r = wc.stride(1);
  // Lambdas, not function pointers: device code cannot make indirect calls.
  switch (qtype) {
    case TYPE_Q3_K:
      run_moe(queue, xp, wp, ip, yp, pairs, rows, top_k, in_features, nblk, bsize, stride_e, stride_r,
              [](const uint8_t *b, const float *v) { return dot_q3_k(b, v); });
      break;
    case TYPE_Q4_K:
      run_moe(queue, xp, wp, ip, yp, pairs, rows, top_k, in_features, nblk, bsize, stride_e, stride_r,
              [](const uint8_t *b, const float *v) { return dot_q4_k(b, v); });
      break;
    case TYPE_Q5_K:
      run_moe(queue, xp, wp, ip, yp, pairs, rows, top_k, in_features, nblk, bsize, stride_e, stride_r,
              [](const uint8_t *b, const float *v) { return dot_q5_k(b, v); });
      break;
    case TYPE_Q6_K:
      run_moe(queue, xp, wp, ip, yp, pairs, rows, top_k, in_features, nblk, bsize, stride_e, stride_r,
              [](const uint8_t *b, const float *v) { return dot_q6_k(b, v); });
      break;
  }
  return y.to(out_dtype);
}

}  // namespace

TORCH_LIBRARY(gguf_xpu, m) {
  m.def("moe_vec(Tensor x, Tensor w, Tensor ids, int top_k, int qtype, int rows) -> Tensor");
}

TORCH_LIBRARY_IMPL(gguf_xpu, XPU, m) {
  m.impl("moe_vec", &moe_vec);
}
