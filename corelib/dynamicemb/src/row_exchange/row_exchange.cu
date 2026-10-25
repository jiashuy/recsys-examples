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
// A cache node holds a copy of the value row of some key. Giving the node to a new key
// is two moves of data that touch the same node: the old key's row, which the cache may
// have trained, goes back to the row that key has in the storage; and the new key's row,
// if it has one, comes from the storage into the node.
//
// Over a link, a GPU pays a fixed price for every random row it reaches in host memory,
// whatever the width of the row, and reaches rows in order about twice as fast. The
// victims are therefore gathered from the cache (in HBM) into a staging buffer first, in
// the order of the storage rows they go back to, which also frees their nodes; and one
// kernel takes item i of the two lists, each sorted by its own storage row: the i-th
// victim goes out to its row and the i-th fetched row goes into its node, both directions
// at once, each in order. The pairing of a node's old row with its new one is gone from
// the kernel because the staging buffer has already kept the old row.
//
// Rows are copied as raw words of 16, 4 or 2 bytes, whichever divides the byte width of
// every table's row; the element type does not matter to a copy.
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

// Item i of two lists, each sorted by its own storage row: the staging row i (a victim)
// goes out to storage row evict_rows[i] of table evict_tids[i], and the storage row
// fetch_rows[i] of table fetch_tids[i] goes into node fetch_nodes[i] of the cache. The two
// need not be of the same table, nor of the same width: each side copies its own. A row of
// -1 skips a side.
template <typename Word>
__global__ void swap_staged_kernel(
    int64_t n, const int64_t *__restrict__ cache_ptrs,
    const int64_t *__restrict__ store_ptrs, const int64_t *__restrict__ row_bytes,
    const int64_t *__restrict__ evict_tids, const int64_t *__restrict__ fetch_tids,
    const int64_t *__restrict__ evict_rows, const int64_t *__restrict__ fetch_rows,
    const int64_t *__restrict__ fetch_nodes, const Word *__restrict__ staging,
    int64_t stage_words) {
  for (int64_t i = blockIdx.x; i < n; i += gridDim.x) {
    const int64_t evict = evict_rows[i];
    const int64_t fetch = fetch_rows[i];
    if (evict >= 0) {
      const int64_t t = evict_tids[i];
      const int64_t words = row_bytes[t] / static_cast<int64_t>(sizeof(Word));
      Word *out = reinterpret_cast<Word *>(store_ptrs[t]) + evict * words;
      const Word *stage = staging + i * stage_words;
      for (int64_t c = threadIdx.x; c < words; c += blockDim.x) out[c] = stage[c];
    }
    if (fetch >= 0) {
      const int64_t t = fetch_tids[i];
      const int64_t words = row_bytes[t] / static_cast<int64_t>(sizeof(Word));
      const Word *in = reinterpret_cast<const Word *>(store_ptrs[t]) + fetch * words;
      Word *node =
          reinterpret_cast<Word *>(cache_ptrs[t]) + fetch_nodes[i] * words;
      for (int64_t c = threadIdx.x; c < words; c += blockDim.x) node[c] = in[c];
    }
  }
}

void check_i64_cuda(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.defined() && t.is_cuda() && t.is_contiguous() &&
                  t.scalar_type() == at::kLong,
              name, " must be a contiguous int64 CUDA tensor");
}

// swap_staged_kernel over n items; `staging` is n rows of stage_bytes each, victims in the
// order of evict_rows. cache_ptrs and store_ptrs are the per-table base pointers of the two
// sets of buffers; row_bytes the width in bytes of a row of each table, the same in both.
void swap_staged(at::Tensor cache_ptrs, at::Tensor store_ptrs, at::Tensor row_bytes,
                 at::Tensor evict_tids, at::Tensor fetch_tids, at::Tensor evict_rows,
                 at::Tensor fetch_rows, at::Tensor fetch_nodes, at::Tensor staging,
                 int64_t stage_bytes, int64_t word_bytes) {
  for (const auto &p : {std::make_pair(&cache_ptrs, "cache_ptrs"),
                        std::make_pair(&store_ptrs, "store_ptrs"),
                        std::make_pair(&row_bytes, "row_bytes"),
                        std::make_pair(&evict_tids, "evict_tids"),
                        std::make_pair(&fetch_tids, "fetch_tids"),
                        std::make_pair(&evict_rows, "evict_rows"),
                        std::make_pair(&fetch_rows, "fetch_rows"),
                        std::make_pair(&fetch_nodes, "fetch_nodes")}) {
    check_i64_cuda(*p.first, p.second);
  }
  const int64_t n = evict_rows.numel();
  TORCH_CHECK(fetch_rows.numel() == n && evict_tids.numel() == n &&
                  fetch_tids.numel() == n && fetch_nodes.numel() == n,
              "one table and row per item, on each side");
  TORCH_CHECK(staging.is_cuda() && staging.is_contiguous() &&
                  staging.numel() * staging.element_size() >= n * stage_bytes &&
                  stage_bytes % word_bytes == 0,
              "staging must be a CUDA tensor of n rows of stage_bytes");
  if (n == 0) return;
  const c10::cuda::CUDAGuard guard(evict_rows.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t grid = std::min<int64_t>(n, 65535);
  auto launch = [&](auto tag) {
    using Word = decltype(tag);
    swap_staged_kernel<Word><<<grid, kBlock, 0, stream>>>(
        n, cache_ptrs.data_ptr<int64_t>(), store_ptrs.data_ptr<int64_t>(),
        row_bytes.data_ptr<int64_t>(), evict_tids.data_ptr<int64_t>(),
        fetch_tids.data_ptr<int64_t>(), evict_rows.data_ptr<int64_t>(),
        fetch_rows.data_ptr<int64_t>(), fetch_nodes.data_ptr<int64_t>(),
        reinterpret_cast<const Word *>(staging.data_ptr()),
        stage_bytes / static_cast<int64_t>(sizeof(Word)));
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
  m.def("swap_staged", &dyn_emb::swap_staged,
        "swap the victims staged for a cache with storage rows, the two sides each "
        "sorted by storage row",
        py::arg("cache_ptrs"), py::arg("store_ptrs"), py::arg("row_bytes"),
        py::arg("evict_tids"), py::arg("fetch_tids"), py::arg("evict_rows"),
        py::arg("fetch_rows"), py::arg("fetch_nodes"), py::arg("staging"),
        py::arg("stage_bytes"), py::arg("word_bytes"));
}
