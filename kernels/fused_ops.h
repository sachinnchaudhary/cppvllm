#pragma once

#include <ATen/ATen.h>

namespace fused {

void rms_norm(at::Tensor& out, const at::Tensor& x, const at::Tensor& w, double eps);

void add_rms_norm(at::Tensor& resid, const at::Tensor& x, const at::Tensor& w, at::Tensor& out, double eps);

void qk_norm_rope_kv_write_fp8(at::Tensor& q_out, const at::Tensor& qkv, const at::Tensor& q_norm, const at::Tensor& k_norm,
                               const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& positions,
                               const at::Tensor& slots, at::Tensor& k_cache, at::Tensor& v_cache, at::Tensor& k_scale,
                               at::Tensor& v_scale, at::Tensor& k_dense, at::Tensor& v_dense, int64_t n_heads,
                               int64_t n_kv_heads, double eps);

void swiglu(at::Tensor& out, const at::Tensor& gate_up);

void sample_tokens(at::Tensor& out, const at::Tensor& logits, const at::Tensor& rows, const at::Tensor& temps,
                   uint64_t seed, uint64_t offset);
                   
void feed_tokens(at::Tensor& tokens, const at::Tensor& src, const at::Tensor& prev);
}  