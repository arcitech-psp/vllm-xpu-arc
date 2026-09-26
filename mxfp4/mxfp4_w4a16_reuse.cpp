// Candidate-local W4A16 operator with a per-thread persistent oneDNN scratchpad.
// The primary vllm_xpu_mxfp4_woq operator is intentionally left untouched.
#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace {
using namespace dnnl;

struct Primitive {
  engine eng;
  memory::desc a, w, s, y;
  matmul::primitive_desc pd;
  matmul op;
  at::Tensor scratch;

  Primitive(sycl::queue& q, int64_t m, int64_t n, int64_t k,
            memory::data_type type, memory::data_type scale_type)
      : eng(sycl_interop::make_engine(q.get_device(), q.get_context())),
        a({m, k}, type, {k, 1}),
        w({k, n}, memory::data_type::f4_e2m1, {1, k}),
        s({k / 32, n}, scale_type, {n, 1}),
        y({m, n}, type, {n, 1}) {
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    attr.set_fpmath_mode(type == memory::data_type::f16
                             ? fpmath_mode::f16
                             : fpmath_mode::bf16);
    attr.set_scales(DNNL_ARG_WEIGHTS, 3, {32, 1}, scale_type);
    pd = matmul::primitive_desc(eng, a, w, y, attr);
    op = matmul(pd);
  }
};

torch::Tensor run(const torch::Tensor& a, const torch::Tensor& w,
                  const torch::Tensor& scales) {
  TORCH_CHECK(a.device().is_xpu() && w.device() == a.device() &&
                  scales.device() == a.device(),
              "Same XPU required");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && scales.dim() == 2 &&
                  a.is_contiguous() && w.is_contiguous() &&
                  scales.is_contiguous(),
              "Contiguous rank-two tensors required");
  TORCH_CHECK(a.scalar_type() == at::kHalf ||
                  a.scalar_type() == at::kBFloat16,
              "FP16/BF16 activations required");
  TORCH_CHECK(w.scalar_type() == at::kByte ||
                  w.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
              "Packed original MXFP4 bytes required");
  const auto st = scales.scalar_type();
  TORCH_CHECK(st == at::ScalarType::Float8_e8m0fnu || st == at::kHalf ||
                  st == at::kFloat,
              "Transposed E8M0/FP16/FP32 scales required");
  const auto scale_type = st == at::kHalf
                              ? memory::data_type::f16
                              : st == at::kFloat ? memory::data_type::f32
                                                 : memory::data_type::e8m0;
  const auto m = a.size(0), k = a.size(1), n = w.size(0);
  TORCH_CHECK(m > 0 && n > 0 && k % 32 == 0 && w.size(1) * 2 == k &&
                  scales.size(0) == k / 32 && scales.size(1) == n,
              "Invalid MXFP4 dimensions");
  const at::DeviceGuard guard(a.device());
  auto& q = c10::xpu::getCurrentXPUStream(a.get_device()).queue();
  const auto type = a.scalar_type() == at::kHalf ? memory::data_type::f16
                                                  : memory::data_type::bf16;
  const auto key = std::to_string(a.get_device()) + ":" +
                   std::to_string(m) + ":" + std::to_string(n) + ":" +
                   std::to_string(k) + ":" + std::to_string(static_cast<int>(type)) +
                   ":" + std::to_string(static_cast<int>(scale_type));
  static thread_local std::unordered_map<std::string,
                                         std::unique_ptr<Primitive>> cache;
  auto& entry = cache[key];
  if (!entry)
    entry = std::make_unique<Primitive>(q, m, n, k, type, scale_type);
  auto& p = *entry;

  // The worker uses one ordered XPU stream per thread. Reusing this scratch
  // buffer avoids a device allocation for every linear on every decode step;
  // queue ordering keeps reuse after the previous primitive execution.
  if (!p.scratch.defined()) {
    p.scratch = torch::empty(
        {static_cast<int64_t>(p.pd.scratchpad_desc().get_size())},
        a.options().dtype(at::kByte));
  }
  auto out = torch::empty({m, n}, a.options());
  const auto mem = [&](const memory::desc& md, void* ptr) {
    return sycl_interop::make_memory(md, p.eng,
                                     sycl_interop::memory_kind::usm, ptr);
  };
  std::unordered_map<int, memory> args{
      {DNNL_ARG_SRC, mem(p.a, a.data_ptr())},
      {DNNL_ARG_WEIGHTS, mem(p.w, w.data_ptr())},
      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
       mem(p.s, scales.data_ptr())},
      {DNNL_ARG_DST, mem(p.y, out.data_ptr())},
      {DNNL_ARG_SCRATCHPAD,
       mem(p.pd.scratchpad_desc(), p.scratch.data_ptr())}};
  auto stream = sycl_interop::make_stream(p.eng, q);
  sycl_interop::execute(p.op, stream, args);
  return out;
}
}  // namespace

TORCH_LIBRARY(vllm_xpu_mxfp4_woq_reuse, m) {
  m.def("matmul(Tensor a, Tensor w, Tensor scales) -> Tensor");
}
TORCH_LIBRARY_IMPL(vllm_xpu_mxfp4_woq_reuse, XPU, m) {
  m.impl("matmul", &run);
}
