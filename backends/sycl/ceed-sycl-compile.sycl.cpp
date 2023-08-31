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
#include <level_zero/ze_api.h>

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

  char       *jit_defs_path, *jit_defs_source;
  const char *sycl_jith_path = "ceed/jit-source/sycl/sycl-jit.h";

  // Prepend defined constants
  for (const auto &[name, value] : constants) {
    oss << "#define " << name << " " << value << "\n";
  }

  // libCeed definitions for Sycl Backends
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
  flags = {std::string("-cl-std=CL3.0"), std::string("-Dint32_t=int")};
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compile an OpenCL source to SPIR-V using Intel's online compiler extension
//------------------------------------------------------------------------------
static inline int CeedJitCompileSource_Sycl(Ceed ceed, const sycl::device &sycl_device, const std::string &opencl_source, ByteVector_t &il_binary,
                                            const std::vector<std::string> &flags = {}) {
  sycl::ext::libceed::online_compiler<sycl::ext::libceed::source_language::opencl_c> compiler(sycl_device);

  try {
    il_binary = compiler.compile(opencl_source, flags);
  } catch (sycl::ext::libceed::online_compile_error &e) {
    return CeedError((ceed), CEED_ERROR_BACKEND, e.what());
  }
  return CEED_ERROR_SUCCESS;
}

// ------------------------------------------------------------------------------
// Load (compile) SPIR-V source and wrap in sycl kernel_bundle
// ------------------------------------------------------------------------------
static int CeedLoadModule_Sycl(Ceed ceed, const sycl::context &sycl_context, const sycl::device &sycl_device, const ByteVector_t &il_binary,
                               SyclModule_t **sycl_module) {
  auto lz_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
  auto lz_device  = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  ze_module_desc_t lz_mod_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC,
                                  nullptr,  // extension specific structs
                                  ZE_MODULE_FORMAT_IL_SPIRV,
                                  il_binary.size(),
                                  il_binary.data(),
                                  " -ze-opt-large-register-file",  // flags
                                  nullptr};                        // specialization constants

  ze_module_handle_t           lz_module;
  ze_module_build_log_handle_t lz_log;
  ze_result_t                  lz_err = zeModuleCreate(lz_context, lz_device, &lz_mod_desc, &lz_module, &lz_log);

  if (ZE_RESULT_SUCCESS != lz_err) {
    size_t log_size = 0;
    char  *log_message;

    zeModuleBuildLogGetString(lz_log, &log_size, nullptr);

    CeedCall(CeedCalloc(log_size, &log_message));
    zeModuleBuildLogGetString(lz_log, &log_size, log_message);

    return CeedError(ceed, CEED_ERROR_BACKEND, "Failed to compile Level Zero module:\n%s", log_message);
  }

  // sycl make_<type> only throws errors for backend mismatch--assume we have vetted this already
  *sycl_module = new SyclModule_t(sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>(
      {lz_module, sycl::ext::oneapi::level_zero::ownership::transfer}, sycl_context));
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

  *sycl_kernel = new sycl::kernel(sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {*sycl_module, lz_kernel, sycl::ext::oneapi::level_zero::ownership::transfer}, data->sycl_context));
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
