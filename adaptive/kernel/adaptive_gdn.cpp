// Include the copied implementation; no replacement of the installed library.
#include "gdn_attn/gdn_attn_interface.cpp"
} // namespace vllm_xpu_adaptive

TORCH_LIBRARY(vllm_xpu_adaptive, m) {
  m.def("gdn_attention(Tensor! core_attn_out, Tensor! z, Tensor projected_states_qkvz, Tensor projected_states_ba, int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim, Tensor! conv_state, Tensor! ssm_state, Tensor conv_weights, Tensor? conv_bias, str activation, Tensor A_log, Tensor dt_bias, int num_prefills, int num_decodes, int num_spec_decodes, Tensor? has_initial_state, Tensor? non_spec_query_start_loc, Tensor? non_spec_token_indx, Tensor? non_spec_state_indices_tensor, Tensor? spec_query_start_loc, Tensor? spec_token_indx, Tensor? spec_state_indices_tensor, Tensor? num_accepted_tokens, int num_actual_tokens, int tp_size, bool reorder_input) -> ()");
  m.impl("gdn_attention", torch::kXPU, &vllm_xpu_adaptive::gdn_attention);
}
