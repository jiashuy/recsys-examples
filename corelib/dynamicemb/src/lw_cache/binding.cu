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
// Python bindings for lw_cache.
//
// The cache is bound together with the stub backing, because P0's acceptance is
// a differential test — the same key/update sequence driven through the cache
// and through a plain model, compared afterwards — and that is far easier to
// generate, shrink and iterate on in Python than in C++.
//
// Rows are exposed as torch tensors over the existing allocations, with no
// copy, so the caller does gather/scatter and the optimizer step with ordinary
// torch ops. That keeps the C++ surface to the five operations the cache
// actually owns.
//
// BUILD STATUS: never compiled. No CUDA toolkit on the development machine, so
// nothing in this file has been through nvcc. The layers underneath it are
// exercised by test/lw_cache/run_host_tests.sh; this one is not.
// ---------------------------------------------------------------------------

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>
#include <vector>

#include "lw_cache/launch.cuh"
#include "lw_cache/testing/stub_backing.h"

namespace py = pybind11;

namespace {

using Value = float;
using LwCacheT = lw_cache::Cache<Value>;
using LwBackingT = lw_cache::testing::StubBacking<Value>;
using lw_cache::AllocSlot;
using lw_cache::Key;
using lw_cache::kNoNode;
using lw_cache::Row;

// Fresh keys — those the backing has no record of, whether never seen or
// evicted — need their rows generated rather than transferred. `derive_from_key`
// makes the contents predictable, which is what the differential test wants;
// production plugs in the real initializer here.
struct InitFill {
  Value fill;
  bool derive_from_key;
  __host__ __device__ Value operator()(Key key, int elem) const {
    return derive_from_key ? static_cast<Value>(key) * 100.0f +
                                 static_cast<Value>(elem)
                           : fill;
  }
};

void check_keys(const at::Tensor& keys) {
  TORCH_CHECK(keys.dim() == 1, "lw_cache: keys must be 1-D");
  TORCH_CHECK(keys.scalar_type() == at::kLong, "lw_cache: keys must be int64");
  TORCH_CHECK(keys.is_cuda(), "lw_cache: keys must be on the device");
  TORCH_CHECK(keys.is_contiguous(), "lw_cache: keys must be contiguous");
}

cudaStream_t current_stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

// ---------------------------------------------------------------------------

class PyBacking {
 public:
  PyBacking(int64_t slots, int64_t row_elems)
      : impl_(std::make_unique<LwBackingT>(slots, static_cast<int>(row_elems))) {}

  LwBackingT& impl() { return *impl_; }

  int64_t slots() const { return impl_->slots(); }
  int64_t row_elems() const { return impl_->elems(); }
  int64_t size() const { return impl_->size(); }

  // Rows live in mapped pinned host memory; this is a CPU view of them, no copy.
  at::Tensor rows() {
    return torch::from_blob(
        impl_->row_host(0), {impl_->slots(), impl_->elems()},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  }

  bool contains(Key key) const { return impl_->contains(key); }
  int64_t probe(Key key) const { return impl_->probe_host(key); }

  // Drops keys and rebuilds, so every surviving key changes row. Nothing in the
  // cache stores a row index, so this must be invisible to it — which is
  // precisely why the fixture does it the hard way.
  void evict(const std::vector<Key>& keys) { impl_->evict_host(keys); }

  // Live (key, row) pairs, sorted by key. Slot order is deliberately not part
  // of it: eviction and rebuild move rows around.
  py::tuple snapshot() {
    const auto snap = impl_->snapshot();
    const int64_t n = static_cast<int64_t>(snap.size());
    const int64_t d = impl_->elems();
    auto keys = torch::empty({n}, torch::dtype(torch::kInt64));
    auto rows = torch::empty({n, d}, torch::dtype(torch::kFloat32));
    auto* kp = keys.data_ptr<int64_t>();
    auto* rp = rows.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
      kp[i] = snap[static_cast<size_t>(i)].first;
      for (int64_t e = 0; e < d; ++e) {
        rp[i * d + e] = snap[static_cast<size_t>(i)].second[static_cast<size_t>(e)];
      }
    }
    return py::make_tuple(keys, rows);
  }

 private:
  std::unique_ptr<LwBackingT> impl_;
};

// ---------------------------------------------------------------------------

class PyCache {
 public:
  PyCache(int64_t capacity, int64_t row_elems, const std::string& replacement,
          int64_t clock_max_second_chances, int64_t prefetch_depth,
          int64_t expected_unique_per_step, double capacity_safety_factor,
          double init_fill, bool init_derive_from_key, bool record_metrics)
      : record_metrics_(record_metrics) {
    lw_cache::CacheConfig cfg;
    cfg.capacity = capacity;
    cfg.row_elems = static_cast<int>(row_elems);
    if (replacement == "fifo") {
      cfg.replacement = lw_cache::ReplacementPolicy::kFifo;
    } else if (replacement == "clock") {
      cfg.replacement = lw_cache::ReplacementPolicy::kClock;
    } else {
      TORCH_CHECK(false, "lw_cache: replacement must be 'fifo' or 'clock'");
    }
    cfg.clock_max_second_chances = static_cast<int>(clock_max_second_chances);
    cfg.prefetch_depth = static_cast<int>(prefetch_depth);
    cfg.expected_unique_per_step = expected_unique_per_step;
    cfg.capacity_safety_factor = capacity_safety_factor;

    impl_ = std::make_unique<LwCacheT>(cfg);
    init_ = InitFill{static_cast<Value>(init_fill), init_derive_from_key};

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    scratch_ = torch::zeros({1}, opts);
  }

  int64_t capacity() const { return impl_->capacity(); }
  int64_t row_elems() const { return impl_->row_elems(); }
  int64_t num_buckets() const { return impl_->num_buckets(); }
  int64_t metadata_bytes() const {
    return static_cast<int64_t>(impl_->metadata_bytes());
  }
  int64_t row_bytes() const { return static_cast<int64_t>(impl_->row_bytes()); }

  void reset() { impl_->reset(); }

  // The row pool, no copy. The caller gathers from it and writes the optimizer
  // result back into it with ordinary torch ops.
  at::Tensor rows() {
    return torch::from_blob(
        view().rows, {impl_->capacity(), impl_->row_elems()},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // Keys must already be deduplicated: reference counting is one +1 per unique
  // key per step, and a repeat would add a reference the single release cannot
  // undo.
  at::Tensor lookup(const at::Tensor& keys) {
    check_keys(keys);
    const int64_t n = keys.size(0);
    auto nodes = torch::empty(
        {n}, torch::TensorOptions().dtype(torch::kInt32).device(keys.device()));
    launcher_.lookup_pin(view(), keys.data_ptr<Key>(), n,
                         nodes.data_ptr<int32_t>(), current_stream());
    return nodes;
  }

  // Fills in the misses, in place. `nodes` comes from lookup(); entries that are
  // still kNoNode afterwards could not be cached — either the backing is full
  // or the ring ran dry, and the caller must route those keys through its
  // bypass path rather than ignore them.
  at::Tensor allocate(const at::Tensor& keys, at::Tensor nodes,
                      PyBacking& backing) {
    check_keys(keys);
    TORCH_CHECK(nodes.scalar_type() == at::kInt && nodes.is_cuda(),
                "lw_cache: nodes must be an int32 CUDA tensor");
    TORCH_CHECK(nodes.size(0) == keys.size(0),
                "lw_cache: nodes and keys must have the same length");
    TORCH_CHECK(backing.row_elems() == impl_->row_elems(),
                "lw_cache: backing row width does not match the cache");

    // Compaction runs on the device. The obvious alternative, torch::nonzero,
    // has to bring the count back to the host, and that synchronisation sits in
    // the middle of the allocate path — wasteful at prefetch depth one, and at
    // greater depth it also orders allocate against release, which is what the
    // launcher's stream event is for. Keep the sync and the event never has to
    // be right.
    const int64_t n = keys.size(0);
    auto slot_bytes = torch::empty(
        {static_cast<int64_t>(n * sizeof(AllocSlot))},
        torch::TensorOptions().dtype(torch::kUInt8).device(keys.device()));
    auto* slots = reinterpret_cast<AllocSlot*>(slot_bytes.data_ptr<uint8_t>());
    auto count = torch::zeros(
        {1}, torch::TensorOptions().dtype(torch::kInt32).device(keys.device()));
    auto* d_count = reinterpret_cast<uint32_t*>(count.data_ptr<int32_t>());

    const cudaStream_t stream = current_stream();
    const int grid = lw_cache::grid_for(n);

    lw_cache::compact_misses_kernel<Value>
        <<<grid, lw_cache::kDefaultBlock, 0, stream>>>(
            nodes.data_ptr<int32_t>(), keys.data_ptr<Key>(), n, slots, d_count);
    lw_cache::cuda_check(cudaGetLastError(), "compact_misses_kernel");

    // Read the ring bound after the launcher's wait on the last release, so it
    // cannot include a slot that a pusher reserved but has not yet written.
    uint64_t tail = 0;
    lw_cache::cuda_check(
        cudaMemcpyAsync(&tail, view().ring.tail, sizeof(tail),
                        cudaMemcpyDeviceToHost, stream),
        "read ring tail");
    lw_cache::cuda_check(cudaStreamSynchronize(stream), "sync ring tail");

    launcher_.allocate_compacted(view(), backing.impl().view(), slots, d_count,
                                 n, tail, impl_->config().pop_policy(),
                                 impl_->retry_buffer(), impl_->clock_enabled(),
                                 init_, stream);

    lw_cache::scatter_nodes_kernel<<<grid, lw_cache::kDefaultBlock, 0, stream>>>(
        slots, d_count, nodes.data_ptr<int32_t>());
    lw_cache::cuda_check(cudaGetLastError(), "scatter_nodes_kernel");

    return nodes;
  }

  void release(const at::Tensor& nodes) {
    TORCH_CHECK(nodes.scalar_type() == at::kInt && nodes.is_cuda(),
                "lw_cache: nodes must be an int32 CUDA tensor");
    launcher_.release(view(), nodes.data_ptr<int32_t>(), nodes.size(0),
                      current_stream());
  }

  // Checkpoint. Drains first, then writes every resident row with no filter:
  // skipping any is data loss, since a long-resident hot key has never been
  // written back at all.
  void flush(PyBacking& backing) {
    launcher_.flush(view(), backing.impl().view(), current_stream());
    lw_cache::cuda_check(cudaStreamSynchronize(current_stream()), "flush sync");
  }

  // Is this key resident? Read-only: unlike lookup() it does not pin, so it
  // does not perturb what it observes.
  //
  // Tests need this to reason about backing eviction. A key the backing drops
  // while the cache still holds it has *not* lost its history — the cached copy
  // is the newest one and will be written back to a freshly allocated row
  // (INV-9). Only a key absent from both is genuinely forgotten.
  //
  // O(capacity) on the host. A debug and test API, not a hot path.
  bool contains(Key key) const {
    const auto keys = impl_->download_keys();
    const auto states = impl_->download_states();
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i] != key) continue;
      const auto s = lw_cache::state_of(states[i]);
      if (s == lw_cache::kResident || s == lw_cache::kLoading) return true;
    }
    return false;
  }

  // Zero after a drain. Anything else is a reference leak, which otherwise only
  // shows up as a hit rate that decays with no obvious cause.
  // Cumulative since construction or the last reset. The copy out is queued on
  // the current stream and never waited on by the step path, so read this every
  // few hundred steps rather than every step.
  py::dict metrics() {
    metrics_.snapshot_async(current_stream());
    lw_cache::cuda_check(cudaStreamSynchronize(current_stream()), "metrics sync");
    const auto s = metrics_.read();
    py::dict d;
    for (int i = 0; i < lw_cache::kNumMetrics; ++i) {
      d[py::str(lw_cache::metric_name(i))] = py::int_(s[i]);
    }
    d["hit_rate"] = py::float_(s.hit_rate());
    // What fraction of evictions CLOCK would defer. Valid under FIFO too, which
    // is the point: the decision comes out of an ordinary run.
    d["clock_opportunity"] = py::float_(s.clock_opportunity());
    d["pop_amplification"] = py::float_(s.pop_amplification());
    d["healthy"] = py::bool_(s.healthy());
    return d;
  }

  void reset_metrics() { metrics_.reset_async(current_stream()); }

  int64_t count_referenced() {
    return static_cast<int64_t>(launcher_.count_referenced(
        view(),
        reinterpret_cast<unsigned long long*>(scratch_.data_ptr<int64_t>()),
        current_stream()));
  }

 private:
  // A disabled metrics view turns every counter bump into an untaken,
  // warp-uniform branch, so the hot path pays nothing when this is off.
  lw_cache::CacheView<Value> view() const {
    return impl_->view(record_metrics_ ? metrics_.view()
                                       : lw_cache::MetricsView{nullptr});
  }

  std::unique_ptr<LwCacheT> impl_;
  lw_cache::Launcher launcher_;
  lw_cache::MetricsBuffer metrics_;
  bool record_metrics_ = false;
  InitFill init_{0.0f, false};
  at::Tensor scratch_;
};

// ---------------------------------------------------------------------------
// Multi-table
// ---------------------------------------------------------------------------

// One backing per table. Row widths are per table — that is most of the point
// of splitting the structures — and a backing has a single width, so sharing one
// would push every table back to a common width.
class PyMultiBacking {
 public:
  PyMultiBacking(const std::vector<int64_t>& slots,
                 const std::vector<int64_t>& row_elems) {
    TORCH_CHECK(!slots.empty(), "lw_cache: need at least one backing");
    TORCH_CHECK(slots.size() == row_elems.size(),
                "lw_cache: slots and row_elems must have the same length");
    std::vector<LwBackingT::View> views;
    for (size_t t = 0; t < slots.size(); ++t) {
      tables_.push_back(std::make_unique<LwBackingT>(
          slots[t], static_cast<int>(row_elems[t])));
      views.push_back(tables_.back()->view());
    }
    const size_t bytes = views.size() * sizeof(LwBackingT::View);
    lw_cache::cuda_check(cudaMalloc(&d_views_, bytes), "cudaMalloc backing views");
    lw_cache::cuda_check(
        cudaMemcpy(d_views_, views.data(), bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy backing views");
  }
  ~PyMultiBacking() { cudaFree(d_views_); }

  PyMultiBacking(const PyMultiBacking&) = delete;
  PyMultiBacking& operator=(const PyMultiBacking&) = delete;

  lw_cache::MultiBackingView<LwBackingT::View> view() const {
    return lw_cache::MultiBackingView<LwBackingT::View>{d_views_};
  }

  int64_t num_tables() const { return static_cast<int64_t>(tables_.size()); }
  LwBackingT& table(int64_t t) { return *tables_[static_cast<size_t>(t)]; }

  int64_t size(int64_t t) { return table(t).size(); }
  bool contains(int64_t t, Key key) { return table(t).contains(key); }
  void evict(int64_t t, const std::vector<Key>& keys) {
    table(t).evict_host(keys);
  }

  py::tuple snapshot(int64_t t) {
    auto& b = table(t);
    const auto snap = b.snapshot();
    const int64_t n = static_cast<int64_t>(snap.size());
    const int64_t d = b.elems();
    auto keys = torch::empty({n}, torch::dtype(torch::kInt64));
    auto rows = torch::empty({n, d}, torch::dtype(torch::kFloat32));
    auto* kp = keys.data_ptr<int64_t>();
    auto* rp = rows.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
      kp[i] = snap[static_cast<size_t>(i)].first;
      for (int64_t e = 0; e < d; ++e) {
        rp[i * d + e] = snap[static_cast<size_t>(i)].second[static_cast<size_t>(e)];
      }
    }
    return py::make_tuple(keys, rows);
  }

 private:
  std::vector<std::unique_ptr<LwBackingT>> tables_;
  LwBackingT::View* d_views_ = nullptr;
};

class PyMultiCache {
 public:
  PyMultiCache(const std::vector<int64_t>& capacities,
               const std::vector<int64_t>& row_elems,
               const std::string& replacement, int64_t clock_max_second_chances,
               int64_t prefetch_depth, double init_fill,
               bool init_derive_from_key) {
    TORCH_CHECK(capacities.size() == row_elems.size(),
                "lw_cache: capacities and row_elems must have the same length");
    std::vector<lw_cache::CacheConfig> cfgs;
    for (size_t t = 0; t < capacities.size(); ++t) {
      lw_cache::CacheConfig c;
      c.capacity = capacities[t];
      c.row_elems = static_cast<int>(row_elems[t]);
      if (replacement == "fifo") {
        c.replacement = lw_cache::ReplacementPolicy::kFifo;
      } else if (replacement == "clock") {
        c.replacement = lw_cache::ReplacementPolicy::kClock;
      } else {
        TORCH_CHECK(false, "lw_cache: replacement must be 'fifo' or 'clock'");
      }
      c.clock_max_second_chances = static_cast<int>(clock_max_second_chances);
      c.prefetch_depth = static_cast<int>(prefetch_depth);
      cfgs.push_back(c);
    }
    impl_ = std::make_unique<lw_cache::MultiCache<Value>>(cfgs);
    init_ = InitFill{static_cast<Value>(init_fill), init_derive_from_key};

    // Flat node index -> table, the same shape as table_range but over nodes.
    std::vector<int64_t> node_range(cfgs.size() + 1, 0);
    for (size_t t = 0; t < cfgs.size(); ++t) {
      node_range[t + 1] = node_range[t] + cfgs[t].capacity;
    }
    total_nodes_ = node_range.back();
    auto i64_cuda = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    node_range_ = torch::from_blob(node_range.data(),
                                   {static_cast<int64_t>(node_range.size())},
                                   torch::dtype(torch::kInt64))
                      .clone()
                      .to(torch::kCUDA);
    tails_ = torch::zeros({static_cast<int64_t>(cfgs.size())}, i64_cuda);
    scratch_ = torch::zeros({1}, i64_cuda);
  }

  int64_t num_tables() const { return impl_->num_tables(); }
  int64_t capacity(int64_t t) const { return impl_->table(static_cast<int>(t)).capacity(); }
  int64_t row_elems(int64_t t) const {
    return impl_->table(static_cast<int>(t)).row_elems();
  }
  int64_t metadata_bytes() const {
    return static_cast<int64_t>(impl_->metadata_bytes());
  }
  int64_t row_bytes() const { return static_cast<int64_t>(impl_->row_bytes()); }

  void reset() { impl_->reset(); }

  // Per-table row pool, no copy. Widths differ across tables, which is why
  // there is no single stacked tensor.
  at::Tensor rows(int64_t t) {
    auto& tab = impl_->table(static_cast<int>(t));
    return torch::from_blob(
        tab.view().rows, {tab.capacity(), tab.row_elems()},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // `keys` must be grouped by table and deduplicated within each, which is what
  // segmented_unique produces. `table_range` has num_tables + 1 entries.
  at::Tensor lookup(const at::Tensor& keys, const at::Tensor& table_range) {
    check_keys(keys);
    check_range(table_range);
    const int64_t n = keys.size(0);
    auto nodes = torch::empty(
        {n}, torch::TensorOptions().dtype(torch::kInt32).device(keys.device()));
    launcher_.mt_lookup_pin(impl_->view(), keys.data_ptr<Key>(),
                            table_range.data_ptr<int64_t>(), n,
                            nodes.data_ptr<int32_t>(), current_stream());
    return nodes;
  }

  at::Tensor allocate(const at::Tensor& keys, const at::Tensor& table_range,
                      at::Tensor nodes, PyMultiBacking& backing) {
    check_keys(keys);
    check_range(table_range);
    TORCH_CHECK(nodes.scalar_type() == at::kInt && nodes.is_cuda(),
                "lw_cache: nodes must be an int32 CUDA tensor");
    TORCH_CHECK(backing.num_tables() == num_tables(),
                "lw_cache: one backing per table is required");

    // Device-side compaction, same reasoning as the single-table path: a host
    // sync here would order allocate against release and hide whether the
    // launcher's event does.
    const int64_t n = keys.size(0);
    auto slot_bytes = torch::empty(
        {static_cast<int64_t>(n * sizeof(AllocSlot))},
        torch::TensorOptions().dtype(torch::kUInt8).device(keys.device()));
    auto* slots = reinterpret_cast<AllocSlot*>(slot_bytes.data_ptr<uint8_t>());
    auto count = torch::zeros(
        {1}, torch::TensorOptions().dtype(torch::kInt32).device(keys.device()));
    auto* d_count = reinterpret_cast<uint32_t*>(count.data_ptr<int32_t>());

    const cudaStream_t stream = current_stream();
    const int grid = lw_cache::grid_for(n);

    // The multi-table variant also resolves each miss's table, because after
    // compaction the misses are no longer grouped by table.
    lw_cache::mt_compact_misses_kernel<Value>
        <<<grid, lw_cache::kDefaultBlock, 0, stream>>>(
            nodes.data_ptr<int32_t>(), keys.data_ptr<Key>(),
            table_range.data_ptr<int64_t>(), static_cast<int>(num_tables()), n,
            slots, d_count);
    lw_cache::cuda_check(cudaGetLastError(), "mt_compact_misses_kernel");

    snapshot_tails(stream);

    launcher_.mt_allocate_compacted(
        impl_->view(), backing.view(), slots, d_count, n,
        reinterpret_cast<const uint64_t*>(tails_.data_ptr<int64_t>()),
        impl_->table(0).config().pop_policy(), impl_->clock_enabled(), init_,
        stream);

    lw_cache::scatter_nodes_kernel<<<grid, lw_cache::kDefaultBlock, 0, stream>>>(
        slots, d_count, nodes.data_ptr<int32_t>());
    lw_cache::cuda_check(cudaGetLastError(), "scatter_nodes_kernel");

    return nodes;
  }

  void release(const at::Tensor& nodes, const at::Tensor& table_range) {
    check_range(table_range);
    TORCH_CHECK(nodes.scalar_type() == at::kInt && nodes.is_cuda(),
                "lw_cache: nodes must be an int32 CUDA tensor");
    launcher_.mt_release(impl_->view(), nodes.data_ptr<int32_t>(),
                         table_range.data_ptr<int64_t>(), nodes.size(0),
                         current_stream());
  }

  void flush(PyMultiBacking& backing) {
    launcher_.mt_flush(impl_->view(), backing.view(),
                       node_range_.data_ptr<int64_t>(), total_nodes_,
                       current_stream());
    lw_cache::cuda_check(cudaStreamSynchronize(current_stream()), "flush sync");
  }

  int64_t count_referenced() {
    return static_cast<int64_t>(launcher_.mt_count_referenced(
        impl_->view(), node_range_.data_ptr<int64_t>(), total_nodes_,
        reinterpret_cast<unsigned long long*>(scratch_.data_ptr<int64_t>()),
        current_stream()));
  }

  bool contains(int64_t t, Key key) const {
    const auto& tab = impl_->table(static_cast<int>(t));
    const auto keys = tab.download_keys();
    const auto states = tab.download_states();
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i] != key) continue;
      const auto s = lw_cache::state_of(states[i]);
      if (s == lw_cache::kResident || s == lw_cache::kLoading) return true;
    }
    return false;
  }

 private:
  static void check_range(const at::Tensor& r) {
    TORCH_CHECK(r.dim() == 1 && r.scalar_type() == at::kLong && r.is_cuda(),
                "lw_cache: table_range must be a 1-D int64 CUDA tensor");
  }

  void snapshot_tails(cudaStream_t stream) {
    const int T = static_cast<int>(num_tables());
    std::vector<int64_t> host(T);
    for (int t = 0; t < T; ++t) {
      uint64_t head = 0, tail = 0;
      impl_->table(t).download_ring_counters(&head, &tail);
      host[static_cast<size_t>(t)] = static_cast<int64_t>(tail);
    }
    lw_cache::cuda_check(
        cudaMemcpyAsync(tails_.data_ptr<int64_t>(), host.data(),
                        sizeof(int64_t) * host.size(), cudaMemcpyHostToDevice,
                        stream),
        "upload tail snapshots");
    lw_cache::cuda_check(cudaStreamSynchronize(stream), "sync tail snapshots");
  }

  std::unique_ptr<lw_cache::MultiCache<Value>> impl_;
  lw_cache::Launcher launcher_;
  InitFill init_{0.0f, false};
  at::Tensor node_range_;
  at::Tensor tails_;
  at::Tensor scratch_;
  int64_t total_nodes_ = 0;
};

}  // namespace

void bind_lw_cache(py::module& m) {
  py::class_<PyBacking>(m, "LwStubBacking",
                        "Test-fixture backing: fixed-capacity open-addressed "
                        "table with rows in mapped pinned host memory.")
      .def(py::init<int64_t, int64_t>(), py::arg("slots"), py::arg("row_elems"))
      .def_property_readonly("slots", &PyBacking::slots)
      .def_property_readonly("row_elems", &PyBacking::row_elems)
      .def("size", &PyBacking::size)
      .def("rows", &PyBacking::rows)
      .def("contains", &PyBacking::contains, py::arg("key"))
      .def("probe", &PyBacking::probe, py::arg("key"))
      .def("evict", &PyBacking::evict, py::arg("keys"),
           "Drop keys and rebuild, relocating every surviving row. The cache "
           "must be indifferent to this.")
      .def("snapshot", &PyBacking::snapshot);

  py::class_<PyCache>(m, "LwCache", "GPU embedding cache (training path).")
      .def(py::init<int64_t, int64_t, const std::string&, int64_t, int64_t,
                    int64_t, double, double, bool, bool>(),
           py::arg("capacity"), py::arg("row_elems"),
           py::arg("replacement") = "fifo",
           py::arg("clock_max_second_chances") = 1,
           py::arg("prefetch_depth") = 1,
           py::arg("expected_unique_per_step") = 0,
           py::arg("capacity_safety_factor") = 1.5, py::arg("init_fill") = 0.0,
           py::arg("init_derive_from_key") = false,
           py::arg("record_metrics") = false)
      .def_property_readonly("capacity", &PyCache::capacity)
      .def_property_readonly("row_elems", &PyCache::row_elems)
      .def_property_readonly("num_buckets", &PyCache::num_buckets)
      .def_property_readonly("metadata_bytes", &PyCache::metadata_bytes)
      .def_property_readonly("row_bytes", &PyCache::row_bytes)
      .def("reset", &PyCache::reset)
      .def("rows", &PyCache::rows, "The row pool as a CUDA tensor, no copy.")
      .def("lookup", &PyCache::lookup, py::arg("keys"),
           "Pin the keys that are resident. Keys must be deduplicated.")
      .def("allocate", &PyCache::allocate, py::arg("keys"), py::arg("nodes"),
           py::arg("backing"),
           "Fill in the misses. Entries left at -1 could not be cached and "
           "must go through the caller's bypass path.")
      .def("release", &PyCache::release, py::arg("nodes"))
      .def("flush", &PyCache::flush, py::arg("backing"))
      .def("contains", &PyCache::contains, py::arg("key"),
           "Whether the key is resident. Does not pin. O(capacity); debug only.")
      .def("metrics", &PyCache::metrics,
           "Cumulative counters. 'clock_opportunity' is the fraction of "
           "evictions CLOCK would defer, and is meaningful while running FIFO.")
      .def("reset_metrics", &PyCache::reset_metrics)
      .def("count_referenced", &PyCache::count_referenced);

  py::class_<PyMultiBacking>(
      m, "LwMultiStubBacking",
      "One stub backing per table. Separate rather than shared because row "
      "widths are per table.")
      .def(py::init<const std::vector<int64_t>&, const std::vector<int64_t>&>(),
           py::arg("slots"), py::arg("row_elems"))
      .def_property_readonly("num_tables", &PyMultiBacking::num_tables)
      .def("size", &PyMultiBacking::size, py::arg("table"))
      .def("contains", &PyMultiBacking::contains, py::arg("table"), py::arg("key"))
      .def("evict", &PyMultiBacking::evict, py::arg("table"), py::arg("keys"))
      .def("snapshot", &PyMultiBacking::snapshot, py::arg("table"));

  py::class_<PyMultiCache>(
      m, "LwMultiCache",
      "Several tables through one launch. Each owns a complete set of "
      "structures, so row width, bucket space and ring are all per table.")
      .def(py::init<const std::vector<int64_t>&, const std::vector<int64_t>&,
                    const std::string&, int64_t, int64_t, double, bool>(),
           py::arg("capacities"), py::arg("row_elems"),
           py::arg("replacement") = "fifo",
           py::arg("clock_max_second_chances") = 1,
           py::arg("prefetch_depth") = 1, py::arg("init_fill") = 0.0,
           py::arg("init_derive_from_key") = false)
      .def_property_readonly("num_tables", &PyMultiCache::num_tables)
      .def_property_readonly("metadata_bytes", &PyMultiCache::metadata_bytes)
      .def_property_readonly("row_bytes", &PyMultiCache::row_bytes)
      .def("capacity", &PyMultiCache::capacity, py::arg("table"))
      .def("row_elems", &PyMultiCache::row_elems, py::arg("table"))
      .def("reset", &PyMultiCache::reset)
      .def("rows", &PyMultiCache::rows, py::arg("table"),
           "That table's row pool as a CUDA tensor, no copy. Widths differ "
           "across tables, so there is no single stacked view.")
      .def("lookup", &PyMultiCache::lookup, py::arg("keys"),
           py::arg("table_range"),
           "Keys must be grouped by table and deduplicated within each.")
      .def("allocate", &PyMultiCache::allocate, py::arg("keys"),
           py::arg("table_range"), py::arg("nodes"), py::arg("backing"))
      .def("release", &PyMultiCache::release, py::arg("nodes"),
           py::arg("table_range"))
      .def("flush", &PyMultiCache::flush, py::arg("backing"))
      .def("contains", &PyMultiCache::contains, py::arg("table"), py::arg("key"))
      .def("count_referenced", &PyMultiCache::count_referenced);
}
