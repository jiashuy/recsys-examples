/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include "hstu.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Hstu_fwd_params* params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t target_group_size,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t h_rab,
                      const size_t d,
                      const float alpha,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor rab,
                      const at::Tensor kv_cache,
                      at::Tensor out,
                      void* num_contexts_d,
                      void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d,
                      void* page_offsets,
                      void* page_ids,
                      void* last_page_lens,
                      void* cu_seqlens_t_d,
                      void* num_targets_d,
                      bool has_rab,
                      bool is_delta_q,
                      bool is_paged_kv,
                      int window_size_left,
                      int window_size_right) {
  // Reset the parameters
  *params = {};

  params->arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;

  // Set the pointers and strides.
  params->q_ptr = q.data_ptr();
  params->k_ptr = k.data_ptr();
  params->v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params->q_row_stride = q.stride(-3);
  params->k_row_stride = k.stride(-3);
  params->v_row_stride = v.stride(-3);
  params->q_head_stride = q.stride(-2);
  params->k_head_stride = k.stride(-2);
  params->v_head_stride = v.stride(-2);
  if (out.numel() > 0) {
    params->o_ptr = out.data_ptr();
    params->o_row_stride = out.stride(-3);
    params->o_head_stride = out.stride(-2);
  }

  params->has_rab = has_rab;
  #ifdef HSTU_DISABLE_RAB
      TORCH_CHECK(!has_rab, "This hstu attention build does not support has_rab.");
  #endif
  if (has_rab) {
    params->rab_ptr = rab.data_ptr();
    params->rab_seqlen_qk_stride = rab.stride(-4);
    params->rab_seqlen_q_stride = rab.stride(-3);
    params->rab_seqlen_k_stride = rab.stride(-2);
    params->h_rab = h_rab;
  } else {
    params->rab_ptr = nullptr;
    params->rab_seqlen_qk_stride = 0;
    params->rab_seqlen_q_stride = 0;
    params->rab_seqlen_k_stride = 0;
    params->h_rab = 0;
  }

  params->num_contexts = static_cast<int*>(num_contexts_d);
  params->cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params->cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params->cu_seqlens_t = static_cast<int*>(cu_seqlens_t_d);
  params->num_targets = static_cast<int*>(num_targets_d);

  params->is_bf16 = q.dtype() == torch::kBFloat16;
  #ifdef HSTU_DISABLE_BF16
    TORCH_CHECK(!params->is_bf16, "This hstu attention build does not support bf16.");
  #endif
  #ifdef HSTU_DISABLE_FP16
    TORCH_CHECK(q.dtype() != torch::kFloat16, "This hstu attention build does not support fp16.");
  #endif

  // Set the block scheduling
  // float coeff = 0.3;
  params->is_balance_fwd = false;
  params->is_balance_bwd = false;
  // auto dprops = at::cuda::getCurrentDeviceProperties();
  // int l2_size = dprops->l2CacheSize;
  // int sm_count = dprops->multiProcessorCount;
  // int num_KV = std::min(sm_count, int(b * h_k));
  // int kv_cache_size = 2 * seqlen_k * num_KV * d * sizeof(k.dtype()) * coeff;
  // int do_cache_size = seqlen_q * num_KV * d * sizeof(out.dtype()) * coeff;
  // if (kv_cache_size < l2_size) {
  //   params->is_balance_fwd = true;
  // }
  // if (kv_cache_size + do_cache_size < l2_size) {
  //   params->is_balance_bwd = true;
  // }

  // Set the dimensions.
  params->b = b;
  params->h = h;
  params->h_k = h_k;
  params->h_h_k_ratio = h / h_k;
  params->seqlen_q = seqlen_q;
  params->seqlen_k = seqlen_k;
  params->seqlen_q_rounded = seqlen_q_rounded;
  params->seqlen_k_rounded = seqlen_k_rounded;
  params->d = d;
  params->alpha = alpha;
  // Set the masks.
  params->is_target = (num_targets_d != nullptr) || (cu_seqlens_t_d != nullptr);
  #ifdef HSTU_DISABLE_TARGET
    TORCH_CHECK(!params->is_target, "This hstu attention build does not support target mask.");
  #endif
  params->target_group_size = target_group_size;
  params->target_group_size_inv = 1.0f / target_group_size;
  if (params->is_target) {
    TORCH_CHECK(target_group_size > 0, "target_group_size must be greater than 0 when target is True");
  }
  params->is_context = num_contexts_d != nullptr;
  #ifdef HSTU_DISABLE_CONTEXT
    TORCH_CHECK(!params->is_context, "This hstu attention build does not support context mask.");
  #endif
  params->is_delta_q = is_delta_q;
  #ifdef HSTU_DISABLE_DELTA_Q
    TORCH_CHECK(!params->is_delta_q, "This hstu attention build does not support delta_q.");
  #endif
  if (is_delta_q) {
    TORCH_CHECK(params->seqlen_q <= params->seqlen_k,
                "For delta_q = True, seqlen_q must be less than or equal to seqlen_k.");
    TORCH_CHECK(!params->is_context, "For delta_q = True, context mask must be False.");
  } else {
    TORCH_CHECK(params->seqlen_q == params->seqlen_k,
                "For delta_q = False, seqlen_q must be equal to seqlen_k.");
  }

  if (window_size_left < 0 || window_size_left > (int)seqlen_k) { window_size_left = seqlen_k; }
  if (window_size_right < 0 || window_size_right > (int)seqlen_k) { window_size_right = seqlen_k; }
  params->window_size_left = window_size_left;
  params->window_size_right = window_size_right;

  params->is_causal = params->window_size_left == (int)seqlen_k && params->window_size_right == 0;
  #ifdef HSTU_DISABLE_CAUSAL
    TORCH_CHECK(!params->is_causal, "This hstu attention build does not support causal mask.");
  #endif
  TORCH_CHECK(!(!params->is_causal && params->is_target), "Target mask is True, but causal mask is False, this is undefined behavior.");
  TORCH_CHECK(!(!params->is_causal && params->is_context), "Context mask is True, but causal mask is False, this is undefined behavior.");
  params->is_local = (window_size_left < (int)seqlen_k || window_size_right < (int)seqlen_k) && !params->is_causal;
  #ifdef HSTU_DISABLE_LOCAL
    TORCH_CHECK(!params->is_local, "This hstu attention build does not support local mask.");
  #endif

  params->is_paged_kv = is_paged_kv;
  #ifdef HSTU_DISABLE_PAGED_KV
    TORCH_CHECK(!params->is_paged_kv, "This hstu attention build does not support paged kv.");
  #endif
  if (is_paged_kv) {
    params->kv_cache_ptr         = kv_cache.data_ptr();
    params->kv_cache_row_stride  = kv_cache.stride(-2); // dim
    params->kv_cache_head_stride = kv_cache.stride(-3); // dim * heads
    params->kv_cache_page_stride = kv_cache.stride(-4); // dim * heads * page_size
    params->kv_cache_kvtensor_stride = kv_cache.stride(-5); // dim * heads * page_size * d
    params->page_size = kv_cache.size(-3);
    params->total_pages = kv_cache.size(-5);
  }
  params->page_offsets = static_cast<int*>(page_offsets);
  params->page_ids = static_cast<int*>(page_ids);
  params->last_page_lens = static_cast<int*>(last_page_lens);
}

void set_params_dgrad(Hstu_bwd_params* params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t target_group_size,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t h_rab,
                      const size_t d,
                      const float alpha,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor dout,
                      const at::Tensor rab,
                      const at::Tensor dRab,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      at::Tensor dq_accum,
                      void* num_contexts_d,
                      void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d,
                      void* num_targets_d,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic,
                      bool has_rab,
                      bool has_drab,
                      bool is_delta_q) {
  *params = {};
  set_params_fprop(params, b, seqlen_q, seqlen_k, target_group_size, seqlen_q_rounded,
                   seqlen_k_rounded, h, h_k, h_rab, d, alpha, q, k, v, rab, at::Tensor(),
                   /*out=*/torch::Tensor(),
                   num_contexts_d, cu_seqlens_q_d, cu_seqlens_k_d,
                   nullptr, nullptr, nullptr, nullptr,
                   num_targets_d,
                   has_rab, is_delta_q, false,
                   window_size_left, window_size_right);

  params->has_drab = has_drab;
  #ifdef HSTU_DISABLE_DRAB
    TORCH_CHECK(!has_drab, "This hstu attention build does not support has_drab.");
  #endif
  if (has_rab && has_drab) {
    params->dRab_ptr = dRab.data_ptr();
    params->drab_seqlen_qk_stride = dRab.stride(-4);
    params->drab_seqlen_q_stride = dRab.stride(-3);
    params->drab_seqlen_k_stride = dRab.stride(-2);
  } else {
    params->dRab_ptr = nullptr;
    params->drab_seqlen_qk_stride = 0;
    params->drab_seqlen_q_stride = 0;
    params->drab_seqlen_k_stride = 0;
  }

  // Set the pointers and strides.
  params->do_ptr = dout.data_ptr();
  params->dq_ptr = dq.data_ptr();
  params->dk_ptr = dk.data_ptr();
  params->dv_ptr = dv.data_ptr();

  params->do_row_stride = dout.stride(-3);
  params->do_head_stride = dout.stride(-2);
  params->dq_row_stride = dq.stride(-3);
  params->dk_row_stride = dk.stride(-3);
  params->dv_row_stride = dv.stride(-3);
  params->dq_head_stride = dq.stride(-2);
  params->dk_head_stride = dk.stride(-2);
  params->dv_head_stride = dv.stride(-2);

  params->dq_accum_ptr = dq_accum.data_ptr();
  params->dq_accum_row_stride = dq_accum.stride(-3);
  params->dq_accum_head_stride = dq_accum.stride(-2);

  params->deterministic = deterministic;
  params->dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
}

template <typename Dtype, bool Has_rab, bool Is_local,
          bool Is_causal, bool Is_context, bool Is_target, bool Is_delta_q>
void run_hstu_fwd_headdim(Hstu_fwd_params &params, cudaStream_t stream) {
  ARCH_SWITCH(params.arch, Arch, [&] {
    #ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) { run_hstu_fwd_<Arch, Dtype, 32, Has_rab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) { run_hstu_fwd_<Arch, Dtype, 64, Has_rab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) { run_hstu_fwd_<Arch, Dtype, 128, Has_rab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) { run_hstu_fwd_<Arch, Dtype, 256, Has_rab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
    #endif
  });
}

void run_hstu_fwd(Hstu_fwd_params &params, cudaStream_t stream) {
  RAB_SWITCH(params.has_rab, Has_rab, [&] {
    FP16_BF16_SWITCH(params.is_bf16, [&] {
      #ifndef HSTU_DISABLE_DELTA_Q
      if (params.is_delta_q) {
        #ifndef HSTU_DISABLE_LOCAL
        if (params.is_local) { run_hstu_fwd_headdim<Dtype, Has_rab, true, false, false, false, true>(params, stream); return; }
        #endif
        #ifndef HSTU_DISABLE_TARGET
        if (params.is_target) { run_hstu_fwd_headdim<Dtype, Has_rab, false, true, false, true, true>(params, stream); return; }
        #endif
        run_hstu_fwd_headdim<Dtype, Has_rab, false, true, false, false, true>(params, stream); return;
      }
      #endif
      #ifndef HSTU_DISABLE_LOCAL
      if (params.is_local) { run_hstu_fwd_headdim<Dtype, Has_rab, true, false, false, false, false>(params, stream); return; }
      #endif
      if (!params.is_causal) { run_hstu_fwd_headdim<Dtype, Has_rab, false, false, false, false, false>(params, stream); return; }
      else {
        #ifndef HSTU_DISABLE_CAUSAL
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
          TARGET_SWITCH(params.is_target, Is_target, [&] {
            run_hstu_fwd_headdim<Dtype, Has_rab, false, true, Is_context, Is_target, false>(params, stream);
          });
        });
        #endif
      }
    });
  });
}

std::vector<at::Tensor> hstu_varlen_fwd(
    const at::Tensor& q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    const int max_seqlen_q,
    const int max_seqlen_k,
    std::optional<const at::Tensor>& num_contexts,  // b
    std::optional<const at::Tensor>& num_targets,  // b
    const int target_group_size,
    int window_size_left,
    int window_size_right,
    const float alpha,
    std::optional<at::Tensor>& rab,
    const bool is_delta_q,
    std::optional<const at::Tensor>& kv_cache,
    std::optional<const at::Tensor>& page_offsets,
    std::optional<const at::Tensor>& page_ids,
    std::optional<const at::Tensor>& last_page_lens,
    std::optional<const at::Tensor>& cu_seqlens_t
    ) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dprops->major >= 8, "HSTU only supports Ampere GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "HSTU only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);
  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);

  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(num_heads % num_heads_k == 0,
              "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 128 || head_size == 256,
              "head_size should be 32, 64, 128, or 256");

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_contexts.has_value()) {
    TORCH_CHECK(num_contexts.value().dtype() == torch::kInt32,
                "num_contexts must have dtype int32");
    CHECK_DEVICE(num_contexts.value());
    CHECK_CONTIGUOUS(num_contexts.value());
    CHECK_SHAPE(num_contexts.value(), batch_size);
  }
  if (num_targets.has_value()) {
    TORCH_CHECK(num_targets.value().dtype() == torch::kInt32,
                "num_targets must have dtype int32");
    CHECK_DEVICE(num_targets.value());
    CHECK_CONTIGUOUS(num_targets.value());
    CHECK_SHAPE(num_targets.value(), batch_size);
  }

  at::Tensor out = torch::empty_like(q);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, sizeof(cutlass::uint128_t) / sizeof(q_dtype));
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, sizeof(cutlass::uint128_t) / sizeof(q_dtype));

  bool has_rab = rab.has_value();
  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(rab.value().stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(num_heads == num_heads_rab || num_heads_rab == 1,
                "Number of heads in rab must be 1 or equal to number of heads in query");
    CHECK_SHAPE(rab.value(), batch_size, num_heads_rab, max_seqlen_k, max_seqlen_k);
    if (seqlen_k_rounded != max_seqlen_k) {
      rab = torch::nn::functional::pad(rab.value(), torch::nn::functional::PadFuncOptions(
                           {0, seqlen_k_rounded - max_seqlen_k}));
    }
  }

  bool is_paged_kv = kv_cache.has_value() && page_offsets.has_value() && page_ids.has_value() && last_page_lens.has_value();
  Hstu_fwd_params params;
  set_params_fprop(&params,                  //
                   batch_size,               //
                   max_seqlen_q,             //
                   max_seqlen_k,             //
                   target_group_size,        //
                   seqlen_q_rounded,         //
                   seqlen_k_rounded,         //
                   num_heads,                //
                   num_heads_k,              //
                   num_heads_rab,                 //
                   head_size,                //
                   alpha,                    //
                   q,                        //
                   k,                        //
                   v,                        //
                   has_rab ? rab.value() : torch::Tensor(),              //
                   kv_cache.has_value() ? kv_cache.value() : torch::Tensor(),
                   out,                      //
                   num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,  //
                   cu_seqlens_q.data_ptr(),  //
                   cu_seqlens_k.data_ptr(),  //
                   page_offsets.has_value() ? page_offsets.value().data_ptr() : nullptr, //
                   page_ids.has_value() ? page_ids.value().data_ptr() : nullptr, //
                   last_page_lens.has_value() ? last_page_lens.value().data_ptr() : nullptr, //
                   cu_seqlens_t.has_value() ? cu_seqlens_t.value().data_ptr() : nullptr, //
                   num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,  //
                   has_rab,                  //
                   is_delta_q,               //
                   is_paged_kv,
                   window_size_left,         //
                   window_size_right);       //
  if (max_seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_hstu_fwd(params, stream);
  } else {
    out.zero_();
  }

  return {out, has_rab ? rab.value() : torch::Tensor()};
}

template <typename Dtype, bool Has_rab, bool Has_drab, bool Is_local,
          bool Is_causal, bool Is_context, bool Is_target, bool Is_delta_q>
void run_hstu_bwd_headdim(Hstu_bwd_params &params, cudaStream_t stream) {
  #ifndef HSTU_DISABLE_HDIM32
  if (params.d == 32) { run_hstu_bwd_<Dtype, 32, Has_rab, Has_drab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
  #endif
  #ifndef HSTU_DISABLE_HDIM64
  if (params.d == 64) { run_hstu_bwd_<Dtype, 64, Has_rab, Has_drab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
  #endif
  #ifndef HSTU_DISABLE_HDIM128
  if (params.d == 128) { run_hstu_bwd_<Dtype, 128, Has_rab, Has_drab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
  #endif
  #ifndef HSTU_DISABLE_HDIM256
  if (params.d == 256) { run_hstu_bwd_<Dtype, 256, Has_rab, Has_drab, Is_local, Is_causal, Is_context, Is_target, Is_delta_q>(params, stream); }
  #endif
}

void run_hstu_bwd(Hstu_bwd_params &params, cudaStream_t stream) {
  #ifndef HSTU_DISABLE_BACKWARD
  RAB_DRAB_SWITCH(params.has_rab, params.has_drab, Has_rab, Has_drab, [&] {
    FP16_BF16_SWITCH(params.is_bf16, [&] {
      #ifndef HSTU_DISABLE_DELTA_Q
      if (params.is_delta_q) {
        #ifndef HSTU_DISABLE_LOCAL
        if (params.is_local) { run_hstu_bwd_headdim<Dtype, Has_rab, Has_drab, true, false, false, false, true>(params, stream); return; }
        #endif
        run_hstu_bwd_headdim<Dtype, Has_rab, Has_drab, false, true, false, false, true>(params, stream); return;
      }
      #endif
      #ifndef HSTU_DISABLE_LOCAL
      if (params.is_local) { run_hstu_bwd_headdim<Dtype, Has_rab, Has_drab, true, false, false, false, false>(params, stream); return; }
      #endif
      if (!params.is_causal) { run_hstu_bwd_headdim<Dtype, Has_rab, Has_drab, false, false, false, false, false>(params, stream); return; }
      else {
        #ifndef HSTU_DISABLE_CAUSAL
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
          TARGET_SWITCH(params.is_target, Is_target, [&] {
            run_hstu_bwd_headdim<Dtype, Has_rab, Has_drab, false, true, Is_context, Is_target, false>(params, stream);
          });
        });
        #endif
      }
    });
  });
  #endif
}

std::vector<at::Tensor> hstu_varlen_bwd(
    const at::Tensor& dout,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<const at::Tensor> &dq_,
    std::optional<const at::Tensor> &dk_,
    std::optional<const at::Tensor> &dv_,
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    const int max_seqlen_q,
    const int max_seqlen_k,
    std::optional<const at::Tensor>& num_contexts,  // b
    std::optional<const at::Tensor>& num_targets,  // b
    const int target_group_size,
    int window_size_left,
    int window_size_right,
    const float alpha,
    std::optional<at::Tensor>& rab,
    const bool has_drab,
    const bool is_delta_q,
    const bool deterministic) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dprops->major >= 8, "HSTU only supports Ampere GPUs or newer.");
  TORCH_CHECK(dprops->major == 8 && dprops->minor == 0, "HSTU backward does not support sm86 or sm89.");
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "HSTU only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);
  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);

  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  CHECK_SHAPE(dout, total_q, num_heads, head_size);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 128 || head_size == 256,
              "head_size should be 32, 64, 128, or 256");
  TORCH_CHECK(num_heads % num_heads_k == 0,
              "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, sizeof(cutlass::uint128_t) / sizeof(q_dtype));
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, sizeof(cutlass::uint128_t) / sizeof(q_dtype));

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_contexts.has_value()) {
    TORCH_CHECK(num_contexts.value().dtype() == torch::kInt32,
                "num_contexts must have dtype int32");
    CHECK_DEVICE(num_contexts.value());
    CHECK_CONTIGUOUS(num_contexts.value());
    CHECK_SHAPE(num_contexts.value(), batch_size);
  }
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_targets.has_value()) {
    TORCH_CHECK(num_targets.value().dtype() == torch::kInt32,
                "num_targets must have dtype int32");
    CHECK_DEVICE(num_targets.value());
    CHECK_CONTIGUOUS(num_targets.value());
    CHECK_SHAPE(num_targets.value(), batch_size);
  }

  bool has_rab = rab.has_value();
  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(rab.value().stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(num_heads == num_heads_rab || num_heads_rab == 1,
                "Number of heads in rab must be 1 or equal to number of heads in query");
    CHECK_SHAPE(rab.value(), batch_size, num_heads_rab, max_seqlen_k, seqlen_k_rounded);
  }

  at::Tensor dq = dq_.has_value() ? dq_.value() : torch::empty_like(q);
  at::Tensor dk = dk_.has_value() ? dk_.value() : torch::empty_like(k);
  at::Tensor dv = dv_.has_value() ? dv_.value() : torch::empty_like(v);

  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.get_device()};

  auto opts = q.options();
  at::Tensor dq_accum;
  // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded,
  // num_heads, head_size) because that would be too large if there is a
  // very long sequence and the rest of the sequences are short. Instead, we
  // allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size).
  // Note that 128 is the max block size on the seqlen_q dimension. For dQ,
  // the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
  // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and
  // (i + 1)-th sequence will be at least 128 apart.
  // It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
  // allowed to do. So we won't have to do any bound checking, and performance
  // should stay the same.
  if (!deterministic) {
    dq_accum =
        torch::zeros({total_q + 128 * batch_size, num_heads, head_size},
                     opts.dtype(at::kFloat));
  } else {
    const int nsplits =
        (dprops->multiProcessorCount + batch_size * num_heads - 1) /
        (batch_size * num_heads);
    dq_accum = torch::zeros(
        {nsplits, total_q + 128 * batch_size, num_heads, head_size},
        opts.dtype(at::kFloat));
  }

  Hstu_bwd_params params;
  at::Tensor dRab;
  if (has_drab) {
    TORCH_CHECK(has_rab, "rab must exist when using has_drab");
    /*
    Due to the demand for unequal lengths of q and k, and the customer's desire to support large and complete rab bias,
    it is directly defined here as the size of (max_seqlen_k, seqlen_k_rounded) to be compatible with both equal and unequal lengths of q and kv.
    Some tiles will not write back so we give zeros.
    */
    dRab = torch::zeros({batch_size, num_heads, max_seqlen_k, seqlen_k_rounded}, opts);
  }

  set_params_dgrad(&params,
                   batch_size,               //
                   max_seqlen_q,             //
                   max_seqlen_k,             //
                   target_group_size,        //
                   seqlen_q_rounded,         //
                   seqlen_k_rounded,         //
                   num_heads,                //
                   num_heads_k,              //
                   num_heads_rab,                 //
                   head_size,                //
                   alpha,                    //
                   q,                        //
                   k,                        //
                   v,                        //
                   dout,                     //
                   has_rab ? rab.value() : torch::Tensor(),              //
                   dRab,                     //
                   dq,                       //
                   dk,                       //
                   dv,                       //
                   dq_accum,                 //
                   num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,  //
                   cu_seqlens_q.data_ptr(),  //
                   cu_seqlens_k.data_ptr(),  //
                   num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,  //
                   window_size_left,         //
                   window_size_right,        //
                   deterministic,            //
                   has_rab,                  //
                   has_drab,                 //
                   is_delta_q);              //

  if (max_seqlen_q > 0) {
    run_hstu_bwd(params, stream);
  } else {
    // If max_seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
    dk.zero_();
    dv.zero_();
    if(has_drab) {
      dRab.zero_();
    }
  }

  if (has_drab && seqlen_k_rounded != max_seqlen_k) {
    dRab = dRab.index({"...", torch::indexing::Slice(torch::indexing::None, max_seqlen_k)});
  }

  return { dq, dk, dv, dRab };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "HstuAttention";
  m.def("varlen_fwd", &hstu_varlen_fwd, "Varlen hstu forward pass");
  m.def("varlen_bwd", &hstu_varlen_bwd, "Varlen hstu backward pass");
}
