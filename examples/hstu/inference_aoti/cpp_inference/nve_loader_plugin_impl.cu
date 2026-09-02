/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nve_loader_plugin.h"

#include <cstdio>
#include <exception>
#include <memory>

// Only the 26.05 compatibility build pins a version; default follows NVE.
#if defined(NVE_VERSION)
#if NVE_VERSION != 2605
#error "Only the NVE 26.05 compatibility plugin sets NVE_VERSION"
#endif
#define RECSYS_NVE_2605_COMPAT
#endif

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {

void set_error(
    char* error, std::size_t error_size, const char* message) noexcept {
  if (error != nullptr && error_size > 0) {
    std::snprintf(error, error_size, "%s", message);
  }
}

struct PluginState {
#if !defined(RECSYS_NVE_2605_COMPAT)
  std::shared_ptr<nve::ResourceDirectory> resources;
#endif
  std::unique_ptr<nve::LayerDirectory> layers;
};

}  // namespace

extern "C" RECSYS_NVE_EXPORT void* recsys_nve_loader_create_state(
    const char* package_dir,
    void* aoti_loader_or_null,
    int device_index,
    char* error,
    std::size_t error_size) noexcept {
  if (error != nullptr && error_size > 0) {
    error[0] = '\0';
  }
  try {
#if defined(RECSYS_NVE_2605_COMPAT)
    if (aoti_loader_or_null != nullptr) {
      set_error(error, error_size, "NVE 26.05 state must be created before AOTI");
      return nullptr;
    }
#else
    if (aoti_loader_or_null == nullptr) {
      set_error(error, error_size, "NVE state requires an AOTI loader");
      return nullptr;
    }
#endif
    auto state = std::make_unique<PluginState>();
#if defined(RECSYS_NVE_2605_COMPAT)
    state->layers =
        std::make_unique<nve::LayerDirectory>(package_dir, device_index);
#else
    auto* loader = static_cast<torch::inductor::AOTIModelPackageLoader*>(
        aoti_loader_or_null);
    state->resources = std::make_shared<nve::ResourceDirectory>();
    state->layers = std::make_unique<nve::LayerDirectory>(
        package_dir, *loader, device_index, state->resources);
#endif
    return state.release();
  } catch (const std::exception& exception) {
    set_error(error, error_size, exception.what());
  } catch (...) {
    set_error(error, error_size, "unknown exception during NVE state creation");
  }
  return nullptr;
}

extern "C" RECSYS_NVE_EXPORT void recsys_nve_loader_destroy_state(
    void* state) noexcept {
  delete static_cast<PluginState*>(state);
}
