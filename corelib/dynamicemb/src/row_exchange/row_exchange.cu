/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
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

// ---------------------------------------------------------------------------
// Swapping the rows of a cache with the rows of the storage behind it.
//
// A cache node holds a copy of the value row of some key. Giving the node to a new
// key is two moves of data that touch the same node: the old key's row, which the
// cache may have trained, goes back to the row that key has in the storage; and the
// new key's row, if it has one, comes from the storage into the node. The two have
// to be done in that order for the node -- read what it holds before it is overwritten
// -- and nothing else orders them. So a block takes one node and does both: every
// thread reads its columns of the old row and of the new one, and only then writes
// them. No staging buffer is needed, and because the storage is often in host memory
// the write-back and the fetch go over the link in opposite directions at the same
// time.
//
// Rows are copied as raw words of 16, 4 or 2 bytes, whichever divides the byte width
// of every table's row; the element type does not matter to a copy.
// ---------------------------------------------------------------------------

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "check.h"

namespace py = pybind11;

namespace dyn_emb {
namespace {

constexpr int kBlock = 128;

template <typename Word>
__global__ void swap_rows_kernel(int64_t n, const int64_t *__restrict__ cache_ptrs,
                                 const int64_t *__restrict__ store_ptrs,
                                 const int64_t *__restrict__ row_bytes,
                                 const int64_t *__restrict__ table_ids,
                                 const int64_t *__restrict__ nodes,
                                 const int64_t *__restrict__ evict_rows,
                                 const int64_t *__restrict__ fetch_rows) {
  for (int64_t i = blockIdx.x; i < n; i += gridDim.x) {
    const int64_t node = nodes[i];
    const int64_t evict = evict_rows[i];
    const int64_t fetch = fetch_rows[i];
    if (node < 0 || (evict < 0 && fetch < 0)) continue;

    const int64_t t = table_ids[i];
    const int64_t words = row_bytes[t] / static_cast<int64_t>(sizeof(Word));
    Word *cache = reinterpret_cast<Word *>(cache_ptrs[t]) + node * words;
    Word *store = reinterpret_cast<Word *>(store_ptrs[t]);
    Word *out = evict >= 0 ? store + evict * words : nullptr;
    const Word *in = fetch >= 0 ? store + fetch * words : nullptr;

    for (int64_t c = threadIdx.x; c < words; c += blockDim.x) {
      Word old, incoming;
      if (out != nullptr) old = cache[c];
      if (in != nullptr) incoming = in[c];
      if (out != nullptr) out[c] = old;
      if (in != nullptr) cache[c] = incoming;
    }
  }
}

void check_i64_cuda(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.defined() && t.is_cuda() && t.is_contiguous() &&
                  t.scalar_type() == at::kLong,
              name, " must be a contiguous int64 CUDA tensor");
}

// For the i-th (node, table): the node's row goes to the storage row evict_rows[i],
// and the storage row fetch_rows[i] comes into the node; either is skipped when it
// is -1. cache_ptrs and store_ptrs are the per-table base pointers of the two sets of
// buffers; row_bytes the width in bytes of a row of each table, the same in both.
void swap_rows(at::Tensor cache_ptrs, at::Tensor store_ptrs, at::Tensor row_bytes,
               at::Tensor table_ids, at::Tensor nodes, at::Tensor evict_rows,
               at::Tensor fetch_rows, int64_t word_bytes) {
  for (const auto &p : {std::make_pair(&cache_ptrs, "cache_ptrs"),
                        std::make_pair(&store_ptrs, "store_ptrs"),
                        std::make_pair(&row_bytes, "row_bytes"),
                        std::make_pair(&table_ids, "table_ids"),
                        std::make_pair(&nodes, "nodes"),
                        std::make_pair(&evict_rows, "evict_rows"),
                        std::make_pair(&fetch_rows, "fetch_rows")}) {
    check_i64_cuda(*p.first, p.second);
  }
  const int64_t n = nodes.numel();
  TORCH_CHECK(table_ids.numel() == n && evict_rows.numel() == n &&
                  fetch_rows.numel() == n,
              "one table, one evict row and one fetch row per node");
  if (n == 0) return;
  const c10::cuda::CUDAGuard guard(nodes.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t grid = std::min<int64_t>(n, 65535);

  auto launch = [&](auto tag) {
    using Word = decltype(tag);
    swap_rows_kernel<Word><<<grid, kBlock, 0, stream>>>(
        n, cache_ptrs.data_ptr<int64_t>(), store_ptrs.data_ptr<int64_t>(),
        row_bytes.data_ptr<int64_t>(), table_ids.data_ptr<int64_t>(),
        nodes.data_ptr<int64_t>(), evict_rows.data_ptr<int64_t>(),
        fetch_rows.data_ptr<int64_t>());
  };
  switch (word_bytes) {
  case 16: launch(uint4{}); break;
  case 4: launch(uint32_t{}); break;
  case 2: launch(uint16_t{}); break;
  default: TORCH_CHECK(false, "word_bytes is 16, 4 or 2");
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace
} // namespace dyn_emb

void bind_row_exchange(py::module &m) {
  m.def("swap_rows", &dyn_emb::swap_rows,
        "swap cache rows with storage rows, node by node", py::arg("cache_ptrs"),
        py::arg("store_ptrs"), py::arg("row_bytes"), py::arg("table_ids"),
        py::arg("nodes"), py::arg("evict_rows"), py::arg("fetch_rows"),
        py::arg("word_bytes"));
}
