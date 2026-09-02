/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define RECSYS_NVE_EXPORT __attribute__((visibility("default")))
#else
#define RECSYS_NVE_EXPORT
#endif

extern "C" {

RECSYS_NVE_EXPORT void* recsys_nve_loader_create_state(
    const char* package_dir,
    void* aoti_loader_or_null,
    int device_index,
    char* error,
    std::size_t error_size) noexcept;
RECSYS_NVE_EXPORT void recsys_nve_loader_destroy_state(void* state)
    noexcept;

}  // extern "C"

namespace recsys::nve_loader {

class NveLoaderPlugin {
 public:
  explicit NveLoaderPlugin(std::string package_dir);
  ~NveLoaderPlugin();

  NveLoaderPlugin(const NveLoaderPlugin&) = delete;
  NveLoaderPlugin& operator=(const NveLoaderPlugin&) = delete;

  const char* selected_version() const noexcept;
  bool requires_aoti_loader() const noexcept;
  void create_state(void* aoti_loader_or_null, int device_index);

 private:
  enum class NveVersion {
    kNve2605,
    kDefault,  // Submodule-backed NVE 26.06 and later.
  };

  using CreateStateFn = decltype(&recsys_nve_loader_create_state);
  using DestroyStateFn = decltype(&recsys_nve_loader_destroy_state);

  std::string package_dir_;
  NveVersion version_ = NveVersion::kDefault;
  CreateStateFn create_state_fn_ = nullptr;
  DestroyStateFn destroy_state_fn_ = nullptr;
  void* state_ = nullptr;
};

}  // namespace recsys::nve_loader
