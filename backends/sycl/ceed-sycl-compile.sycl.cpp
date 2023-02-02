// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-compile.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>
#include <ze_api.h>

#include <map>
#include <sstream>
#include <sycl/ext/intel/experimental/online_compiler.hpp>
#include <sycl/sycl.hpp>

#include "ceed-sycl-common.hpp"

using ByteVector_t = std::vector<unsigned char>;

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
static int CeedJitAddDefinitions_Sycl(Ceed ceed, const std::string &kernel_source, std::string &jit_source,
                                      const std::map<std::string, CeedInt> &constants = {}) {
  std::ostringstream oss;

  // Prepend defined constants
  for (const auto &[name, value] : constants) {
    oss << "#define " << name << " " << value << "\n";
  }

  // libCeed definitions for Sycl Backends
  char       *jit_defs_path, *jit_defs_source;
  const char *sycl_jith_path = "ceed/jit-source/sycl/sycl-jit.h";
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, sycl_jith_path, &jit_defs_path));
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, jit_defs_path, &jit_defs_source));

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
  flags = {std::string("-cl-std=CL3.0")};
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compile an OpenCL source to SPIR-V using Intel's online compiler extension
//------------------------------------------------------------------------------
static inline int CeedJitCompileSource_Sycl(Ceed ceed, const sycl::device &sycl_device, const std::string &opencl_source, ByteVector_t &il_binary,
                                            const std::vector<std::string> &flags = {}) {
  sycl::ext::intel::experimental::online_compiler<sycl::ext::intel::experimental::source_language::opencl_c> compiler(sycl_device);

  try {
    il_binary = compiler.compile(opencl_source, flags);
  } catch (sycl::ext::intel::experimental::online_compile_error &e) {
    return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Load (compile) SPIR-V source and wrap in sycl kernel_bundle
// TODO: determine appropriate flags
// TODO: Error handle lz calls
// ------------------------------------------------------------------------------
static int CeedJitLoadModule_Sycl(const sycl::context &sycl_context, const sycl::device &sycl_device, const ByteVector_t &il_binary,
                                  [[maybe_unused]] SyclModule_t *sycl_module) {
  auto lz_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
  auto lz_device  = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  ze_module_desc_t lz_mod_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                  nullptr,
                                  ZE_MODULE_FORMAT_IL_SPIRV,
                                  il_binary.size(),
                                  il_binary.data(),
                                  nullptr,   // flags
                                  nullptr};  // build log

  ze_module_handle_t lz_module;
  zeModuleCreate(lz_context, lz_device, &lz_mod_desc, &lz_module, nullptr);

  // sycl make_<type> only throws errors for backend mismatch--assume we have vetted this already
  sycl_module = new SyclModule_t(sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>(
      {lz_module, sycl::ext::oneapi::level_zero::ownership::transfer}, sycl_context));

  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Compile kernel source to an executable `sycl::kernel_bundle`
// ------------------------------------------------------------------------------
int CeedJitBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t *sycl_module, const std::map<std::string, CeedInt> &constants) {
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  std::string jit_source;
  CeedCallBackend(CeedJitAddDefinitions_Sycl(ceed, kernel_source, jit_source, constants));

  std::vector<std::string> flags;
  CeedCallBackend(CeedJitGetFlags_Sycl(flags));

  ByteVector_t il_binary;
  CeedCallBackend(CeedJitCompileSource_Sycl(ceed, data->sycl_device, jit_source, il_binary, flags));

  CeedCallBackend(CeedJitLoadModule_Sycl(data->sycl_context, data->sycl_device, il_binary, sycl_module));

  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Get a sycl kernel from an existing kernel_bundle
//
// TODO: Error handle lz calls
// ------------------------------------------------------------------------------
int CeedJitGetKernel_Sycl(Ceed ceed, const SyclModule_t *sycl_module, const std::string &kernel_name, [[maybe_unused]] sycl::kernel *sycl_kernel) {
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // sycl::get_native returns std::vector<ze_module_handle_t> for lz backend
  // https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md
  ze_module_handle_t lz_module = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*sycl_module).front();

  ze_kernel_desc_t   lz_kernel_desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernel_name.c_str()};
  ze_kernel_handle_t lz_kernel;
  zeKernelCreate(lz_module, &lz_kernel_desc, &lz_kernel);

  sycl_kernel = new sycl::kernel(sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {*sycl_module, lz_kernel, sycl::ext::oneapi::level_zero::ownership::transfer}, data->sycl_context));

  return CEED_ERROR_SUCCESS;
}
