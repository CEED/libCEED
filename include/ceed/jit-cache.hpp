// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for the caches of the NVRTC and hipRTC JiT compilations of the CUDA and HIP backends
#pragma once

#include <ceed/backend.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

//------------------------------------------------------------------------------
// Cache of JiT compiled binaries
//
// Applications often JiT compile the same kernel source many times, e.g. when they rebuild operators with new data.
// Successful compilations are kept in memory, for the lifetime of the process, keyed by the compiler options and the
// source. The files that the source includes, such as the QFunction source, are assumed not to change while the process
// runs.
//------------------------------------------------------------------------------

/// A JiT compilation, looked up with @ref CeedJitCacheFind() and stored with @ref CeedJitCacheStore()
struct CeedJitCacheEntry {
  std::string key;  // compiler options and source
};

// In internal namespace to avoid polluting files that include this
namespace ceed {
namespace internal {

// State of the cache, shared by all Ceed contexts
struct JitCacheState {
  std::mutex                                   mutex;
  std::unordered_map<std::string, std::string> binaries;  // by key
};

// The state is never destroyed, so that it outlives any use at exit
static inline JitCacheState &GetJitCacheState() {
  static JitCacheState *state = new JitCacheState;
  return *state;
}

}  // namespace internal
}  // namespace ceed

/**
  @brief Look up a JiT compilation in memory

  If `binary` is set to `NULL`, the compiled binary must be stored with @ref CeedJitCacheStore().

  @param[in]  ceed     `Ceed` context
  @param[in]  num_opts Number of compiler options
  @param[in]  opts     Compiler options
  @param[in]  source   Source to compile
  @param[in]  name     Name of the kernel
  @param[out] entry    Lookup state, to pass to @ref CeedJitCacheStore()
  @param[out] binary   Cached binary, or `NULL` if not found

  @return An error code: 0 - success, otherwise - failure
**/
static inline int CeedJitCacheFind(Ceed ceed, int num_opts, const char *const *opts, const std::string &source, const char *name,
                                   CeedJitCacheEntry *entry, const std::string **binary) {
  ceed::internal::JitCacheState &state = ceed::internal::GetJitCacheState();

  *binary = nullptr;
  for (int i = 0; i < num_opts; i++) entry->key.append(opts[i]).push_back('\0');
  entry->key += source;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto                  cached = state.binaries.find(entry->key);

    if (cached != state.binaries.end()) *binary = &cached->second;
  }
  if (*binary) CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- USING CACHED JiT BINARY: %s ----------\n", name);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Store a successful JiT compilation not found by @ref CeedJitCacheFind()

  @param[in]     ceed   `Ceed` context
  @param[in,out] entry  Lookup state from @ref CeedJitCacheFind()
  @param[in]     binary Compiled binary
  @param[in]     size   Size of the compiled binary

  @return An error code: 0 - success, otherwise - failure
**/
static inline int CeedJitCacheStore(Ceed ceed, CeedJitCacheEntry *entry, const char *binary, size_t size) {
  ceed::internal::JitCacheState &state = ceed::internal::GetJitCacheState();
  std::lock_guard<std::mutex>    lock(state.mutex);

  state.binaries.emplace(std::move(entry->key), std::string(binary, size));
  return CEED_ERROR_SUCCESS;
}
