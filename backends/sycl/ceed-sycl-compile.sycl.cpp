// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-compile.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <level_zero/ze_api.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <sycl/sycl.hpp>

#include "./online_compiler.hpp"
#include "ceed-sycl-common.hpp"

using ByteVector_t = std::vector<unsigned char>;

//------------------------------------------------------------------------------
// Add defined constants at the beginning of kernel source
//------------------------------------------------------------------------------
static int CeedJitAddDefinitions_Sycl(Ceed ceed, const std::string &kernel_source, std::string &jit_source,
                                      const std::map<std::string, CeedInt> &constants = {}) {
  std::ostringstream oss;

  const char *jit_defs_path, *jit_defs_source;
  const char *sycl_jith_path = "ceed/jit-source/sycl/sycl-jit.h";

  // Prepend defined constants
  for (const auto &[name, value] : constants) {
    oss << "#define " << name << " " << value << "\n";
  }

  // libCeed definitions for Sycl Backends
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, sycl_jith_path, &jit_defs_path));
  {
    char *source;

    CeedCallBackend(CeedLoadSourceToBuffer(ceed, jit_defs_path, &source));
    jit_defs_source = source;
  }

  oss << jit_defs_source << "\n";

  CeedCallBackend(CeedFree(&jit_defs_path));
  CeedCallBackend(CeedFree(&jit_defs_source));

  // Append kernel_source
  oss << "\n" << kernel_source;

  jit_source = oss.str();
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// TODO: Add architecture flags, optimization flags
//------------------------------------------------------------------------------
static inline int CeedJitGetFlags_Sycl(std::vector<std::string> &flags) {
  flags = {std::string("-cl-std=CL3.0"), std::string("-Dint32_t=int"), std::string("-DCEED_RUNNING_JIT_PASS=1")};
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compute a cache key (hex string) for OpenCL C source + flags
//------------------------------------------------------------------------------
static std::string CeedSpvCacheHash(const std::string &opencl_source, const std::vector<std::string> &flags) {
  size_t h = std::hash<std::string>{}(opencl_source);
  for (const auto &f : flags) {
    h ^= std::hash<std::string>{}(f) + 0x9e3779b9u + (h << 6) + (h >> 2);
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  return oss.str();
}

//------------------------------------------------------------------------------
// Return path to the SPIR-V cache directory (same base as LZ cache).
//------------------------------------------------------------------------------
static std::filesystem::path CeedSpvCacheDir() {
  const char *env = std::getenv("SYCL_CACHE_DIR");
  std::string base;
  if (env && *env) {
    base = env;
  } else {
    const char *home = std::getenv("HOME");
    base              = home ? std::string(home) + "/.cache" : "/tmp";
  }
  return std::filesystem::path(base) / "ceed_spirv";
}

//------------------------------------------------------------------------------
// Compile an OpenCL source to SPIR-V using Intel's online compiler extension.
// Caches the resulting SPIR-V binary to avoid recompilation on subsequent runs.
//------------------------------------------------------------------------------
static inline int CeedJitCompileSource_Sycl(Ceed ceed, const sycl::device &sycl_device, const std::string &opencl_source, ByteVector_t &il_binary,
                                            const std::vector<std::string> &flags = {}) {
  // Check SPIR-V cache first
  std::filesystem::path cache_path;
  try {
    std::filesystem::path cache_dir = CeedSpvCacheDir();
    std::filesystem::create_directories(cache_dir);
    cache_path = cache_dir / (CeedSpvCacheHash(opencl_source, flags) + ".spv");
    if (std::filesystem::exists(cache_path)) {
      std::ifstream f(cache_path, std::ios::binary);
      il_binary.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
      if (!il_binary.empty()) return CEED_ERROR_SUCCESS;
    }
  } catch (...) {
  }

  sycl::ext::libceed::online_compiler<sycl::ext::libceed::source_language::opencl_c> compiler(sycl_device);

  try {
    il_binary = compiler.compile(opencl_source, flags);
  } catch (sycl::ext::libceed::online_compile_error &e) {
    return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }

  // Save SPIR-V to cache
  if (!cache_path.empty() && !il_binary.empty()) {
    try {
      std::ofstream f(cache_path, std::ios::binary);
      f.write(reinterpret_cast<const char *>(il_binary.data()), static_cast<std::streamsize>(il_binary.size()));
    } catch (...) {
    }
  }
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Compute a cache key (hex string) for SPIR-V binary + build flags
// ------------------------------------------------------------------------------
static std::string CeedLzCacheHash(const ByteVector_t &il_binary, const std::string &flags) {
  size_t h = std::hash<std::string>{}(flags);
  for (unsigned char b : il_binary) {
    h ^= std::hash<unsigned char>{}(b) + 0x9e3779b9u + (h << 6) + (h >> 2);
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << h;
  return oss.str();
}

// ------------------------------------------------------------------------------
// Return path to the Level Zero native binary cache directory.
// Uses $SYCL_CACHE_DIR/ceed_lz or $HOME/.cache/ceed_sycl/lz.
// ------------------------------------------------------------------------------
static std::filesystem::path CeedLzCacheDir() {
  const char *env = std::getenv("SYCL_CACHE_DIR");
  std::string base;
  if (env && *env) {
    base = env;
  } else {
    const char *home = std::getenv("HOME");
    base              = home ? std::string(home) + "/.cache" : "/tmp";
  }
  return std::filesystem::path(base) / "ceed_lz";
}

// ------------------------------------------------------------------------------
// Load (compile) SPIR-V source and wrap in sycl kernel_bundle.
// Caches the compiled native GPU binary so subsequent runs skip JIT.
// ------------------------------------------------------------------------------
static int CeedLoadModule_Sycl(Ceed ceed, const sycl::context &sycl_context, const sycl::device &sycl_device, const ByteVector_t &il_binary,
                               SyclModule_t **sycl_module) {
  auto lz_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
  auto lz_device  = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  const std::string build_flags = " -ze-opt-large-register-file";

  // --- Cache lookup ---
  std::filesystem::path cache_path;
  bool                  have_cache = false;
  ByteVector_t          native_binary;

  try {
    std::filesystem::path cache_dir = CeedLzCacheDir();
    std::filesystem::create_directories(cache_dir);
    cache_path = cache_dir / (CeedLzCacheHash(il_binary, build_flags) + ".native");
    if (std::filesystem::exists(cache_path)) {
      std::ifstream f(cache_path, std::ios::binary);
      native_binary.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
      have_cache = !native_binary.empty();
    }
  } catch (...) {
  }

  ze_module_handle_t           lz_module;
  ze_module_build_log_handle_t lz_log;
  ze_result_t                  lz_err;

  if (have_cache) {
    // Load precompiled native binary — skips JIT entirely
    ze_module_desc_t lz_mod_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_NATIVE,
                                    native_binary.size(), native_binary.data(), nullptr, nullptr};
    lz_err = zeModuleCreate(lz_context, lz_device, &lz_mod_desc, &lz_module, &lz_log);
  } else {
    // JIT compile SPIR-V → native
    ze_module_desc_t lz_mod_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr, ZE_MODULE_FORMAT_IL_SPIRV,
                                    il_binary.size(), il_binary.data(), build_flags.c_str(), nullptr};
    lz_err = zeModuleCreate(lz_context, lz_device, &lz_mod_desc, &lz_module, &lz_log);

    // Save native binary to cache for future runs
    if (lz_err == ZE_RESULT_SUCCESS && !cache_path.empty()) {
      size_t native_size = 0;
      if (zeModuleGetNativeBinary(lz_module, &native_size, nullptr) == ZE_RESULT_SUCCESS && native_size > 0) {
        std::vector<uint8_t> out(native_size);
        if (zeModuleGetNativeBinary(lz_module, &native_size, out.data()) == ZE_RESULT_SUCCESS) {
          try {
            std::ofstream f(cache_path, std::ios::binary);
            f.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(native_size));
          } catch (...) {
          }  // cache write failure is non-fatal
        }
      }
    }
  }

  if (ZE_RESULT_SUCCESS != lz_err) {
    size_t log_size = 0;
    char  *log_message;

    zeModuleBuildLogGetString(lz_log, &log_size, nullptr);

    CeedCallBackend(CeedCalloc(log_size, &log_message));
    zeModuleBuildLogGetString(lz_log, &log_size, log_message);

    return CeedError(ceed, CEED_ERROR_BACKEND, "Failed to compile Level Zero module:\n%s", log_message);
  }

  // sycl make_<type> only throws errors for backend mismatch--assume we have vetted this already
  *sycl_module = new SyclModule_t(sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>(
      {lz_module, sycl::ext::oneapi::level_zero::ownership::transfer}, sycl_context));
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Build a kernel_bundle<executable> from a kernel_bundle<input>, with native
// binary caching.  cache_key_extra encodes any specialization constants so
// different specializations get distinct cache entries.
// ------------------------------------------------------------------------------
int CeedBuildBundleCached_Sycl(Ceed ceed, sycl::kernel_bundle<sycl::bundle_state::input> &input_bundle, SyclModule_t **sycl_module,
                                const std::string &cache_key_extra) {
  // Note: native binary caching via zeModuleCreate + make_kernel_bundle does not
  // preserve SYCL kernel IDs for bundles built with specialization constants,
  // causing "kernel bundle does not contain the kernel" at dispatch time.
  // Use sycl::build directly — it is fast since the input bundle is already compiled.
  try {
    *sycl_module = new SyclModule_t(sycl::build(input_bundle));
  } catch (sycl::exception &e) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "sycl::build failed: %s", e.what());
  }
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Compile kernel source to an executable `sycl::kernel_bundle`
// ------------------------------------------------------------------------------
int CeedBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t **sycl_module, const std::map<std::string, CeedInt> &constants) {
  Ceed_Sycl               *data;
  std::string              jit_source;
  std::vector<std::string> flags;
  ByteVector_t             il_binary;

  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedJitAddDefinitions_Sycl(ceed, kernel_source, jit_source, constants));
  CeedCallBackend(CeedJitGetFlags_Sycl(flags));
  CeedCallBackend(CeedJitCompileSource_Sycl(ceed, data->sycl_device, jit_source, il_binary, flags));
  CeedCallBackend(CeedLoadModule_Sycl(ceed, data->sycl_context, data->sycl_device, il_binary, sycl_module));
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Get a sycl kernel from an existing kernel_bundle
//
// TODO: Error handle lz calls
// ------------------------------------------------------------------------------
int CeedGetKernel_Sycl(Ceed ceed, const SyclModule_t *sycl_module, const std::string &kernel_name, sycl::kernel **sycl_kernel) {
  Ceed_Sycl *data;

  CeedCallBackend(CeedGetData(ceed, &data));

  // sycl::get_native returns std::vector<ze_module_handle_t> for lz backend
  // https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md
  ze_module_handle_t lz_module = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*sycl_module).front();

  ze_kernel_desc_t   lz_kernel_desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernel_name.c_str()};
  ze_kernel_handle_t lz_kernel;
  ze_result_t        lz_err = zeKernelCreate(lz_module, &lz_kernel_desc, &lz_kernel);

  if (ZE_RESULT_SUCCESS != lz_err) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "Failed to retrieve kernel from Level Zero module");
  }

  *sycl_kernel = new sycl::kernel(sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>({*sycl_module, lz_kernel,
                                                                                           sycl::ext::oneapi::level_zero::ownership::transfer},
                                                                                          data->sycl_context));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run SYCL kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
int CeedRunKernelDimSharedSycl(Ceed ceed, sycl::kernel *kernel, const int grid_size, const int block_size_x, const int block_size_y,
                               const int block_size_z, const int shared_mem_size, void **kernel_args) {
  sycl::range<3>    local_range(block_size_z, block_size_y, block_size_x);
  sycl::range<3>    global_range(grid_size * block_size_z, block_size_y, block_size_x);
  sycl::nd_range<3> kernel_range(global_range, local_range);

  //-----------
  // Order queue
  Ceed_Sycl *ceed_Sycl;

  CeedCallBackend(CeedGetData(ceed, &ceed_Sycl));
  sycl::event e = ceed_Sycl->sycl_queue.ext_oneapi_submit_barrier();

  ceed_Sycl->sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.set_args(*kernel_args);
    cgh.parallel_for(kernel_range, *kernel);
  });
  return CEED_ERROR_SUCCESS;
}
