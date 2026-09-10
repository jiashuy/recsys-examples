/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// talos_observer.cpp — C++ RecordFunction observer for torch_perf_analysis.
//
// One callback registered at the ATen dispatcher (at::addGlobalCallback)
// emits every NVTX label the analysis pipeline consumes. Per-op work never
// enters the Python interpreter; the only Python-adjacent step is a C-API
// read of the calling thread's own (frozen) frame chain to resolve the user
// call site, memoized in a global hash table keyed on (code object, bytecode
// offset).
//
// Every label is a registered string pushed by handle, with the autograd seq
// carried in the 64-bit event payload (exported as NVTX_EVENTS.int64Value).
//
// Emission (all gated by g_on, set by start()/stop()):
//   * outermost FUNCTION-scope op on a Python thread:
//       range "talos::<op>#<path>@<Class>!!<file>:<line>"   payload = own seq
//     (the #ctx part comes from a C++ thread_local updated by the Python
//      module hooks)
//   * nested FUNCTION-scope op with seq >= 0:
//       mark  "talos_seq:<op>"                              payload = seq
//     — the seq→label join key the backward synthesis uses
//   * autograd::engine::evaluate_function ranges: pushed verbatim
//   * BACKWARD_FUNCTION scope (grad node apply):
//       range "<Node>"                                      payload = seq
//
// The callback is global, so it fires on every thread — framework worker
// threads and the C++ autograd threads included — with no threading patch.

#include <torch/extension.h>
#include <ATen/record_function.h>
#include <nvtx3/nvToolsExt.h>

#include <Python.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ── switches ────────────────────────────────────────────────────────
std::atomic<bool> g_on{false};        // capture window open
std::atomic<bool> g_siteline{true};   // resolve file:line (ablation switch)
std::atomic<bool> g_seq_marks{true};  // emit talos_seq: marks   (ablation switch)

// ── per-thread state ────────────────────────────────────────────────
thread_local std::string t_ctx;  // "path@Class" from the module hooks
thread_local int t_depth = 0;    // FUNCTION-scope nesting depth

// ── no-kernel op skip list ──────────────────────────────────────────
// Outermost ops that never (or almost never) launch device work: CUDA
// allocator bookkeeping and pure-metadata view ops. Labeling them buys no
// attribution (no kernel can fall inside) and costs the full per-range
// price — on the ur191 replay, record_stream ALONE was 43% of all
// outermost ranges. Sync-relevant ops (item/to/contiguous/reshape — they
// copy or synchronize) are deliberately NOT here. Names are matched after
// stripping "aten::".
bool op_is_skipped(const char* op) {
  static const std::unordered_map<std::string, bool> skip = [] {
    std::unordered_map<std::string, bool> s;
    for (const char* n :
         {"record_stream", "transpose",  "t",           "permute",
          "view",          "_unsafe_view", "expand",    "expand_as",
          "slice",         "select",     "narrow",      "unsqueeze",
          "squeeze",       "flatten",    "unflatten",   "as_strided",
          "alias",         "detach",     "detach_",     "lift_fresh",
          "empty",         "empty_like", "empty_strided", "numel",
          "size",          "stride",     "is_contiguous", "sym_size",
          "set_",          "resize_"}) {
      s.emplace(n, true);
    }
    return s;
  }();
  return skip.count(op) != 0;
}

std::atomic<bool> g_use_skiplist{true};

// ── internal-file prefixes (torch/, stdlib, this package) ──────────
std::vector<std::string> g_internal_prefixes;
std::shared_mutex g_prefix_mu;

// Working directory (with trailing sep) — call sites under it are labeled
// with the cwd-relative path, '/' replaced by '.' (veloq joins nvtx_path
// segments with '/', so a slash inside a label would corrupt path parsing):
//   /root/vlog_pt_replay/ur191_pytorch_copy/comm/dnn.py → comm.dnn.py
// Files outside the cwd keep the bare basename.
std::string g_cwd;  // set once via set_cwd() before start(); then read-only

// code object → is-internal (code objects are INCREF-pinned on insert so
// the pointer key can never be recycled under us)
std::unordered_map<void*, bool> g_internal_code;
std::shared_mutex g_internal_mu;

// (code, lasti) → "file.py:123"
struct SiteKey {
  void* code;
  int lasti;
  bool operator==(const SiteKey& o) const {
    return code == o.code && lasti == o.lasti;
  }
};
struct SiteKeyHash {
  size_t operator()(const SiteKey& k) const {
    return std::hash<void*>()(k.code) ^
           (static_cast<size_t>(k.lasti) * 0x9e3779b97f4a7c15ULL);
  }
};
std::unordered_map<SiteKey, std::string, SiteKeyHash> g_site_cache;
std::shared_mutex g_site_mu;

// ── NVTX domain + registered strings ───────────────────────────────
// With nsys attached, every *unique* message string is interned (hashed +
// copied into the report string table) per event, so a label that varied per
// event would make interning the dominant per-event cost. Each distinct label
// is therefore registered ONCE (nvtxDomainRegisterStringA) and pushed by
// handle, with the per-event seq travelling in the 64-bit payload instead.
nvtxDomainHandle_t g_domain = nullptr;  // created at first start()

// full label (without seq) → registered handle
std::unordered_map<std::string, nvtxStringHandle_t> g_reg_cache;
std::shared_mutex g_reg_mu2;

nvtxStringHandle_t reg_string(const char* s) {
  {
    std::shared_lock<std::shared_mutex> lk(g_reg_mu2);
    auto it = g_reg_cache.find(s);
    if (it != g_reg_cache.end()) return it->second;
  }
  std::unique_lock<std::shared_mutex> lk(g_reg_mu2);
  auto it = g_reg_cache.find(s);
  if (it != g_reg_cache.end()) return it->second;
  nvtxStringHandle_t h = nvtxDomainRegisterStringA(g_domain, s);
  g_reg_cache.emplace(s, h);
  return h;
}

// Push a range / emit a mark with a registered message + optional seq
// payload. Message uniqueness no longer scales with event count.
inline void push_reg(const char* label, int64_t seq) {
  nvtxEventAttributes_t a = {};
  a.version = NVTX_VERSION;
  a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  a.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  a.message.registered = reg_string(label);
  if (seq >= 0) {
    a.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    a.payload.llValue = seq;
  }
  nvtxDomainRangePushEx(g_domain, &a);
}

inline void mark_reg(const char* label, int64_t seq) {
  nvtxEventAttributes_t a = {};
  a.version = NVTX_VERSION;
  a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  a.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  a.message.registered = reg_string(label);
  if (seq >= 0) {
    a.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    a.payload.llValue = seq;
  }
  nvtxDomainMarkEx(g_domain, &a);
}

// ── debug counters ──────────────────────────────────────────────────
std::atomic<uint64_t> g_n_ranges{0}, g_n_marks{0}, g_n_bwd{0},
    g_n_site_miss{0}, g_n_nosite{0}, g_n_skipped{0};

// ── call-site resolution (caller must hold the GIL / be attached) ──

bool code_is_internal(PyCodeObject* co) {
  {
    std::shared_lock<std::shared_mutex> lk(g_internal_mu);
    auto it = g_internal_code.find(static_cast<void*>(co));
    if (it != g_internal_code.end()) return it->second;
  }
  bool internal = true;  // anything unreadable: skip past it
  PyObject* fn = PyObject_GetAttrString(reinterpret_cast<PyObject*>(co),
                                        "co_filename");
  if (fn != nullptr) {
    Py_ssize_t len = 0;
    const char* s = PyUnicode_AsUTF8AndSize(fn, &len);
    if (s != nullptr) {
      internal = (len == 0) || s[0] == '<';
      if (!internal) {
        std::shared_lock<std::shared_mutex> lk(g_prefix_mu);
        for (const auto& p : g_internal_prefixes) {
          if (len >= static_cast<Py_ssize_t>(p.size()) &&
              std::memcmp(s, p.data(), p.size()) == 0) {
            internal = true;
            break;
          }
        }
      }
    }
    Py_DECREF(fn);
  } else {
    PyErr_Clear();
  }
  {
    std::unique_lock<std::shared_mutex> lk(g_internal_mu);
    auto ins = g_internal_code.emplace(static_cast<void*>(co), internal);
    if (ins.second) Py_INCREF(reinterpret_cast<PyObject*>(co));  // pin
  }
  return internal;
}

// First non-internal frame above us as a cached "file.py:123" string.
// Returns nullptr when there is none. Storage is stable (node-based map,
// never erased). Caller must be attached to the interpreter.
const std::string* resolve_site() {
  PyFrameObject* f = PyEval_GetFrame();  // borrowed
  if (f == nullptr) return nullptr;
  Py_INCREF(f);
  PyCodeObject* co = PyFrame_GetCode(f);  // new ref
  int guard = 0;
  while (code_is_internal(co)) {
    PyFrameObject* back = PyFrame_GetBack(f);  // new ref
    Py_DECREF(co);
    Py_DECREF(f);
    if (back == nullptr || ++guard > 64) {
      Py_XDECREF(back);
      return nullptr;
    }
    f = back;
    co = PyFrame_GetCode(f);
  }
  SiteKey key{static_cast<void*>(co), PyFrame_GetLasti(f)};
  {
    std::shared_lock<std::shared_mutex> lk(g_site_mu);
    auto it = g_site_cache.find(key);
    if (it != g_site_cache.end()) {
      Py_DECREF(co);
      Py_DECREF(f);
      return &it->second;
    }
  }
  // miss (once per call site): build "basename:line"
  int line = PyFrame_GetLineNumber(f);
  std::string disp = "?";
  PyObject* fnobj = PyObject_GetAttrString(reinterpret_cast<PyObject*>(co),
                                           "co_filename");
  if (fnobj != nullptr) {
    const char* s = PyUnicode_AsUTF8(fnobj);
    if (s != nullptr) {
      if (!g_cwd.empty() &&
          std::strncmp(s, g_cwd.c_str(), g_cwd.size()) == 0) {
        disp = s + g_cwd.size();          // cwd-relative
      } else {
        const char* slash = std::strrchr(s, '/');
        disp = (slash != nullptr) ? slash + 1 : s;  // fallback: basename
      }
      for (auto& c : disp) {
        if (c == '/') c = '.';  // '/' is veloq's nvtx_path separator
      }
    }
  } else {
    PyErr_Clear();
  }
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s:%d", disp.c_str(), line);
  Py_XDECREF(fnobj);
  g_n_site_miss.fetch_add(1, std::memory_order_relaxed);
  const std::string* out;
  {
    std::unique_lock<std::shared_mutex> lk(g_site_mu);
    auto ins = g_site_cache.emplace(key, std::string(buf));
    if (ins.second) Py_INCREF(reinterpret_cast<PyObject*>(co));  // pin
    out = &ins.first->second;
  }
  Py_DECREF(co);
  Py_DECREF(f);
  return out;
}

// ── the observer ────────────────────────────────────────────────────

struct TalosObsCtx : public at::ObserverContext {
  bool pushed = false;
  bool counted = false;
  bool domain_pushed = false;  // pushed via nvtxDomainRangePushEx
};

std::unique_ptr<at::ObserverContext> on_start(const at::RecordFunction& fn) {
  if (!g_on.load(std::memory_order_relaxed)) return nullptr;

  if (fn.scope() == at::RecordScope::BACKWARD_FUNCTION) {
    // grad-node apply — node name registered once, seq in the payload
    auto ctx = std::make_unique<TalosObsCtx>();
    push_reg(fn.name(), fn.seqNr());
    ctx->domain_pushed = true;
    g_n_bwd.fetch_add(1, std::memory_order_relaxed);
    ctx->pushed = true;
    return ctx;
  }

  // FUNCTION scope
  auto ctx = std::make_unique<TalosObsCtx>();
  ctx->counted = true;
  if (t_depth++ > 0) {
    // nested op: only the seq→label join key, one cheap mark
    int64_t seq = fn.seqNr();
    if (seq >= 0 && g_seq_marks.load(std::memory_order_relaxed)) {
      char buf[192];
      std::snprintf(buf, sizeof(buf), "talos_seq:%s", fn.name());
      mark_reg(buf, seq);  // one registration per distinct op name
      g_n_marks.fetch_add(1, std::memory_order_relaxed);
    }
    return ctx;
  }

  const char* name = fn.name();
  if (std::strncmp(name, "autograd::engine::", 18) == 0) {
    push_reg(name, fn.seqNr());  // evaluate_function range
    ctx->domain_pushed = true;
    ctx->pushed = true;
    g_n_bwd.fetch_add(1, std::memory_order_relaxed);
    return ctx;
  }

  const char* op = (std::strncmp(name, "aten::", 6) == 0) ? name + 6 : name;

  if (g_use_skiplist.load(std::memory_order_relaxed) && op_is_skipped(op)) {
    g_n_skipped.fetch_add(1, std::memory_order_relaxed);
    return ctx;  // depth-counted, but no range: op never launches device work
  }

  const std::string* site = nullptr;
  if (g_siteline.load(std::memory_order_relaxed)) {
    // Only threads that already speak Python; never create a tstate for
    // a pure C++ thread (autograd workers).
    PyThreadState* ts = PyGILState_GetThisThreadState();
    if (ts != nullptr) {
      PyGILState_STATE g = PyGILState_Ensure();
      site = resolve_site();
      PyGILState_Release(g);
    }
  }
  if (site == nullptr) g_n_nosite.fetch_add(1, std::memory_order_relaxed);

  char buf[512];
  const char* cs = (site != nullptr) ? site->c_str() : "?:?";
  int64_t seq = fn.seqNr();
  // message is stable per (op, ctx, site) → registered once; seq → payload
  if (!t_ctx.empty()) {
    std::snprintf(buf, sizeof(buf), "talos::%s#%s!!%s", op, t_ctx.c_str(), cs);
  } else {
    std::snprintf(buf, sizeof(buf), "talos::%s!!%s", op, cs);
  }
  push_reg(buf, seq);
  ctx->domain_pushed = true;
  ctx->pushed = true;
  g_n_ranges.fetch_add(1, std::memory_order_relaxed);
  return ctx;
}

void on_end(const at::RecordFunction& /*fn*/, at::ObserverContext* raw) {
  auto* ctx = static_cast<TalosObsCtx*>(raw);
  if (ctx == nullptr) return;
  if (ctx->counted) t_depth--;
  if (ctx->pushed) {
    if (ctx->domain_pushed) {
      nvtxDomainRangePop(g_domain);
    } else {
      nvtxRangePop();
    }
  }
}

// ── registration / bindings ─────────────────────────────────────────

at::CallbackHandle g_handle = 0;
std::mutex g_reg_mu;

void ta_start(bool siteline, bool seq_marks, bool skiplist,
              bool own_domain) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  g_siteline.store(siteline);
  g_seq_marks.store(seq_marks);
  g_use_skiplist.store(skiplist);
  // Default: emit into the DEFAULT NVTX domain (g_domain == nullptr) so our
  // ranges interleave with the workload's own torch.cuda.nvtx regions in
  // one veloq nvtx-path tree. TALOS_CPP_OWN_DOMAIN=1 (via own_domain) keeps a
  // separate "talos" domain instead.
  if (own_domain && g_domain == nullptr)
    g_domain = nvtxDomainCreateA("talos");
  if (g_handle != 0) return;
  g_handle = at::addGlobalCallback(
      at::RecordFunctionCallback(&on_start, &on_end)
          .scopes({at::RecordScope::FUNCTION,
                   at::RecordScope::BACKWARD_FUNCTION}));
  g_on.store(true);
}

void ta_stop() {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  if (g_handle == 0) return;
  g_on.store(false);
  at::removeCallback(g_handle);
  g_handle = 0;
}

void ta_set_ctx(const std::string& s) { t_ctx = s; }

void ta_set_internal_prefixes(const std::vector<std::string>& ps) {
  std::unique_lock<std::shared_mutex> lk(g_prefix_mu);
  g_internal_prefixes = ps;
}

void ta_set_cwd(const std::string& cwd) { g_cwd = cwd; }

pybind11::dict ta_stats() {
  pybind11::dict d;
  d["ranges"] = g_n_ranges.load();
  d["seq_marks"] = g_n_marks.load();
  d["backward"] = g_n_bwd.load();
  d["site_cache_misses"] = g_n_site_miss.load();
  d["no_site"] = g_n_nosite.load();
  d["skipped"] = g_n_skipped.load();
  {
    std::shared_lock<std::shared_mutex> lk(g_site_mu);
    d["site_cache_size"] = g_site_cache.size();
  }
  return d;
}

void ta_bind(pybind11::module_& m) {
  m.def("start", &ta_start, pybind11::arg("siteline") = true,
        pybind11::arg("seq_marks") = true, pybind11::arg("skiplist") = true,
        pybind11::arg("own_domain") = false,
        "Register the global RecordFunction observer and open the gate.");
  m.def("stop", &ta_stop, "Close the gate and unregister the observer.");
  m.def("set_ctx", &ta_set_ctx,
        "Set this thread's module context, e.g. 'Model.dnn@DNN'.");
  m.def("set_internal_prefixes", &ta_set_internal_prefixes,
        "Absolute dir prefixes whose frames are skipped for file:line.");
  m.def("set_cwd", &ta_set_cwd,
        "Working dir (trailing sep); call sites under it get relative paths.");
  m.def("stats", &ta_stats, "Debug counters.");
}

}  // namespace

#ifdef Py_GIL_DISABLED
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, pybind11::mod_gil_not_used()) {
  ta_bind(m);
}
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { ta_bind(m); }
#endif
