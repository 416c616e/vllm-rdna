// pybind11 entry points for the standalone (JIT) build of
// csrc/rocm/gdn_fused_rdna2.cu, used by gdn_fused_bench.py.
#include <torch/extension.h>

void gdn_fused_rdna2(const torch::Tensor& qkvz, const torch::Tensor& ba,
                     const torch::Tensor& conv_w, const torch::Tensor& conv_b,
                     torch::Tensor& conv_state, torch::Tensor& ssm_state,
                     const torch::Tensor& A_log, const torch::Tensor& dt_bias,
                     const torch::Tensor& norm_w, const torch::Tensor& sidx,
                     const torch::Tensor& nacc, const torch::Tensor& qsl,
                     torch::Tensor& y, torch::Tensor& scratch, int64_t num_k_heads,
                     double scale, double eps, int64_t gate_sigmoid);
int64_t gdn_fused_rdna2_scratch_bytes(int64_t M, int64_t HV, int64_t H);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_fused", &gdn_fused_rdna2);
  m.def("scratch_bytes", &gdn_fused_rdna2_scratch_bytes);
}
