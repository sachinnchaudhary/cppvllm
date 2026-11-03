#pragma once  

#include <ATen/ATen.h>  

void paged_attention(at::Tensor& out, const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, 
                     const at::Tensor& block_tables, const at::Tensor& token_seq, const at::Tensor& token_kv_len, double scale);  

                