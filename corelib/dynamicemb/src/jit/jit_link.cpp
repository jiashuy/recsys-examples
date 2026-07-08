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

#include "jit_link.h"

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nvJitLink.h>

namespace dyn_emb {
namespace {

struct EvictModule {
  CUmodule module = nullptr;
  CUfunction ovf = nullptr;
  CUfunction noovf = nullptr;
};

std::mutex g_mu;
std::vector<char> g_lex_fatbin;
bool g_lex_loaded = false;
EvictModule g_lex;                              // key 0 (default evictor)
std::unordered_map<int64_t, EvictModule> g_custom; // key != 0

void cu_check(CUresult r, const char *what) {
  if (r != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("dynamicemb jit_link: ") + what +
                             " failed: " + (msg ? msg : "unknown"));
  }
}

EvictModule load_module(const void *image, const char *ctx) {
  EvictModule m;
  cu_check(cuModuleLoadData(&m.module, image), ctx);
  cu_check(cuModuleGetFunction(&m.ovf, m.module, "dyn_emb_evict_entry_ovf"),
           "cuModuleGetFunction(ovf)");
  cu_check(cuModuleGetFunction(&m.noovf, m.module, "dyn_emb_evict_entry_noovf"),
           "cuModuleGetFunction(noovf)");
  return m;
}

} // namespace

void demb_set_lex_fatbin(const void *lex, size_t lex_size) {
  std::lock_guard<std::mutex> lk(g_mu);
  // Set-once: the default (Lex) evictor is a single packaged fatbin, and the
  // lazily-loaded g_lex module below is not reloaded on a re-set. Ignore repeat
  // calls so the stored bytes never diverge from the loaded module. (Python's
  // ensure_lex_fatbin_loaded() is already idempotent; this guards direct use.)
  if (!g_lex_fatbin.empty())
    return;
  g_lex_fatbin.assign(static_cast<const char *>(lex),
                      static_cast<const char *>(lex) + lex_size);
}

void demb_register_score_function(int64_t key, const void *ltoir,
                                  size_t ltoir_size, const void *cust,
                                  size_t cust_size, int cc_major,
                                  int cc_minor) {
  if (key == 0)
    throw std::runtime_error("dynamicemb jit_link: score_function key 0 is "
                             "reserved for the default evictor");
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_custom.count(key))
    return; // already linked

  char arch[32];
  std::snprintf(arch, sizeof(arch), "-arch=sm_%d%d", cc_major, cc_minor);
  const char *opts[] = {arch, "-lto"};

  nvJitLinkHandle handle;
  auto jl_check = [&](nvJitLinkResult r, const char *what) {
    if (r != NVJITLINK_SUCCESS) {
      std::string log;
      size_t log_size = 0;
      if (nvJitLinkGetErrorLogSize(handle, &log_size) == NVJITLINK_SUCCESS &&
          log_size > 0) {
        log.resize(log_size);
        nvJitLinkGetErrorLog(handle, log.data());
      }
      throw std::runtime_error(std::string("dynamicemb jit_link: ") + what +
                               " failed: " + log);
    }
  };

  if (nvJitLinkCreate(&handle, 2, opts) != NVJITLINK_SUCCESS)
    throw std::runtime_error("dynamicemb jit_link: nvJitLinkCreate failed");
  jl_check(nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, cust, cust_size,
                            "evict_custom"),
           "nvJitLinkAddData(custom fatbin)");
  jl_check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoir, ltoir_size,
                            "user_score_fn"),
           "nvJitLinkAddData(user ltoir)");
  jl_check(nvJitLinkComplete(handle), "nvJitLinkComplete");
  size_t cubin_size = 0;
  jl_check(nvJitLinkGetLinkedCubinSize(handle, &cubin_size),
           "nvJitLinkGetLinkedCubinSize");
  std::vector<char> cubin(cubin_size);
  jl_check(nvJitLinkGetLinkedCubin(handle, cubin.data()),
           "nvJitLinkGetLinkedCubin");
  nvJitLinkDestroy(&handle);

  g_custom[key] = load_module(cubin.data(), "cuModuleLoadData(custom cubin)");
}

CUfunction demb_get_evict_fn(int64_t key, bool overflow) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (key == 0) {
    if (!g_lex_loaded) {
      if (g_lex_fatbin.empty())
        throw std::runtime_error("dynamicemb jit_link: default evict fatbin "
                                 "not set (call demb_set_lex_fatbin first)");
      g_lex = load_module(g_lex_fatbin.data(), "cuModuleLoadData(lex fatbin)");
      g_lex_loaded = true;
    }
    return overflow ? g_lex.ovf : g_lex.noovf;
  }
  auto it = g_custom.find(key);
  if (it == g_custom.end())
    throw std::runtime_error("dynamicemb jit_link: score_function key " +
                             std::to_string(key) + " not registered");
  return overflow ? it->second.ovf : it->second.noovf;
}

void demb_launch_evict(CUfunction fn, EvictParams params, int64_t batch,
                       CUstream stream) {
  constexpr int BLOCK = 256;
  if (batch <= 0)
    return;
  unsigned grid = static_cast<unsigned>((batch + BLOCK - 1) / BLOCK);
  void *args[] = {&params};
  cu_check(cuLaunchKernel(fn, grid, 1, 1, BLOCK, 1, 1, 0, stream, args,
                          nullptr),
           "cuLaunchKernel(evict)");
}

} // namespace dyn_emb
