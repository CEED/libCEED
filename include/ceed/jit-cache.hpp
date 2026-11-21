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

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

// In internal namespace to avoid polluting files that include this
namespace ceed {
namespace internal {

// State for the in-memory cache
struct JitCacheState {
  std::mutex                                                                mutex;
  std::default_random_engine                                                random_generator;
  std::uniform_int_distribution<std::size_t>                                random_distribution;
  std::unordered_map<std::size_t, std::shared_ptr<const std::vector<char>>> binaries;
  std::unordered_map<std::size_t, std::size_t>                              hot_to_disk;  // hash of source -> hash of preprocessed source
};

// The state has static linkage, so it outlives any use of the binaries but gets cleaned up automatically at exit
static inline JitCacheState &GetJitCacheState() {
  static std::random_device r;
  static JitCacheState      state = {{}, std::default_random_engine{r()}, std::uniform_int_distribution<std::size_t>(1000000000), {}, {}};
  return state;
}

/**
  @brief Parse a cache filename

  @param[in]  filename  Filename to parse
  @param[out] hash      Hash of preprocessed source
  @param[out] kernel    Kernel name
  @param[out] extension File extension

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedJitCacheFileGetParts(const std::string &filename, std::size_t *hash, std::string *kernel, std::string *extension) {
  // Filename has the format
  // function_[hash]_[arch]_[kernel].[extension]
  char   function_str[9], kernel_str[1024], extension_str[16];
  size_t hash_local;

  // sscanf to split into tokens:
  // - string excluding '_', length <= 8 ("function")
  // - single char, uncaptured (underscore)
  // - size_t value (hash)
  // - single char, uncaptured (underscore)
  // - string excluding '.', length <= 1024 (kernel name)
  // - single char, uncaptured (period)
  // - string, length <= 10 (extension)
  sscanf(filename.c_str(), "%8[^_]%*c%zu%*c%1023[^.]%*c%15s", function_str, &hash_local, kernel_str, extension_str);
  if (hash) *hash = hash_local;
  if (kernel) *kernel = std::string(kernel_str);
  if (extension) *extension = std::string(extension_str);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the full path for a cache file

  @param[in]  ceed        `Ceed` context
  @param[in]  hash        Hash of preprocessed source
  @param[in]  kernel_name Kernel name
  @param[in]  extension   File extension
  @param[out] path        Location to write the path

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedJitCacheGetPath(Ceed ceed, std::size_t hash, const std::string &kernel_name, const std::string &extension, std::string &path) {
  const char       *cache_dir;
  std::stringstream path_stream;

  CeedCall(CeedGetCacheDir(ceed, &cache_dir));
  path_stream << cache_dir << "/function_" << hash << "_" << kernel_name << extension;
  CeedCall(CeedRestoreCacheDir(ceed, &cache_dir));
  path = path_stream.str();
  return CEED_ERROR_SUCCESS;
}

}  // namespace internal
}  // namespace ceed

/**
  @brief Try to find the given file in the in-memory (hot) cache. Uses hash of the source file, not preprocessed source.

  @param[in]  ceed   `Ceed` context
  @param[in]  source Full source of JiT file, including inserted configuration information
  @param[out] binary Pointer to `const std::vector<char>` in which to store the cached binary, set to `nullptr` if not found
  @param[out] hash   Computed hash of the source file

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedJitCacheFindHot(Ceed ceed, const std::string &source, const std::vector<char> **binary, size_t *hash) {
  // Find in the in-memory map based on the source (and compiler/options) alone
  ceed::internal::JitCacheState &state          = ceed::internal::GetJitCacheState();
  std::size_t                    local_hash     = std::hash<std::string>{}(source);
  const auto                     found_location = state.hot_to_disk.find(local_hash);

  if (hash) *hash = local_hash;
  *binary = NULL;
  if (found_location != state.hot_to_disk.end()) {
    *binary = state.binaries[found_location->second].get();
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- USING CACHED JiT BINARY BASED ON SOURCE HASH: %zu->%zu ----------\n", hash,
                 found_location->second);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Try to find the JiT binary for a preprocessed source, either from the cache or from disk
  @param[in]  ceed                `Ceed` context
  @param[in]  source_hash         Hash of the (not preprocessed) source file
  @param[in]  preprocessed_source Full preprocessed source file
  @param[in]  kernel_name         Kernel name
  @param[in]  extension           File extension for compiled binary
  @param[out] path                Path that the compiled binary is/should be stored at
  @param[out] binary              Pointer to `const std::vector<char>` in which to store the cached binary, set to `nullptr` if not found

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedJitCacheFind(Ceed ceed, std::size_t source_hash, const std::string &preprocessed_source, const std::string &kernel_name,
                            const std::string &extension, std::string &path, const std::vector<char> **binary) {
  // Check the disk cache
  ceed::internal::JitCacheState &state = ceed::internal::GetJitCacheState();
  std::size_t                    hash  = std::hash<std::string>{}(preprocessed_source);
  struct stat                    buffer;

  path.clear();
  const auto found_location = state.binaries.find(hash);
  if (found_location != state.binaries.end()) {
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- USING CACHED JiT BINARY: %s ----------\n", kernel_name.c_str());
    *binary = found_location->second.get();
    return CEED_ERROR_SUCCESS;
  }

  CeedCall(ceed::internal::CeedJitCacheGetPath(ceed, hash, kernel_name, extension, path));
  bool file_exists = (stat(path.c_str(), &buffer) == 0);

  if (file_exists) {
    bool              is_binary = extension == ".cubin" || extension == ".hsaco" || extension == ".so";
    std::ifstream     in_stream = is_binary ? std::ifstream(path, std::ios::binary) : std::ifstream(path);
    std::vector<char> buf(std::istreambuf_iterator<char>(in_stream), {});

    {
      std::lock_guard<std::mutex> lock(state.mutex);

      state.binaries[hash]           = std::make_shared<std::vector<char>>(std::move(buf));
      state.hot_to_disk[source_hash] = hash;
      *binary                        = state.binaries[hash].get();
    }
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- USING JiT BINARY CACHED ON DISK: %s ----------\n", kernel_name.c_str());
    CeedDebug(ceed, "Entry file: %s (hash: %zu)", path.c_str(), hash);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set an entry in the cache

  @param[in]   ceed         `Ceed` context
  @param[in]   source_hash  Hash of (not preprocessed) source file
  @param[in]   path         Path to store cached binary
  @param[out]  binary       Binary to store, pass with `std::move`, ownership is transferred to the cache

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedJitCacheSet(Ceed ceed, std::size_t source_hash, const std::string &path, std::vector<char> &&binary) {
  ceed::internal::JitCacheState &state = ceed::internal::GetJitCacheState();
  std::size_t                    hash;
  std::size_t                    last_slash = path.rfind('/');
  std::string                    filename   = last_slash == std::string::npos ? path : path.substr(last_slash + 1), extension, name;

  CeedCall(ceed::internal::CeedJitCacheFileGetParts(filename, &hash, &name, &extension));
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    struct stat                 buffer;
    bool                        file_exists = (stat(path.c_str(), &buffer) == 0);
    bool                        is_binary   = extension == ".cubin" || extension == ".hsaco" || extension == ".so";

    // Check within the lock to make sure only one thread writes
    if (!file_exists) {
      const std::size_t build_id = state.random_distribution(state.random_generator);
      std::string       filename_tmp;

      {
        const char *dir;

        CeedCallBackend(CeedGetCacheDir(ceed, &dir));
        filename_tmp = std::string(dir) + "/.tmp_function_" + std::to_string(hash) + "_" + name + "_" + std::to_string(build_id) + "." + extension;
        CeedCallBackend(CeedRestoreCacheDir(ceed, &dir));
      }
      std::ofstream out_stream = is_binary ? std::ofstream(filename_tmp, std::ios::binary) : std::ofstream(filename_tmp);

      out_stream.write(binary.data(), binary.size());

      // Atomically try to move to final location
      if (link(filename_tmp.c_str(), path.c_str()) < 0) {
        // EEXIST means another process beat us to it, so succeed silently
        CeedCheck(errno == EEXIST, ceed, CEED_ERROR_BACKEND, "Failed to write '%s' to disk: %s", path.c_str(), strerror(errno));
        errno = 0;
      }

      // Remove temporary file
      {
        int err = errno;

        unlink(filename_tmp.c_str());
        errno = err;
      }

      // Add to in-memory cache
      state.binaries[hash]           = std::make_shared<std::vector<char>>(binary);
      state.hot_to_disk[source_hash] = hash;
    }
  }
  return CEED_ERROR_SUCCESS;
}
