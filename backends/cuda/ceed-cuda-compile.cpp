// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-compile.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <nvrtc.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "ceed-cuda-common.h"

#define CeedChk_Nvrtc(ceed, x)                                                                              \
  do {                                                                                                      \
    nvrtcResult result = static_cast<nvrtcResult>(x);                                                       \
    if (result != NVRTC_SUCCESS) return CeedError((ceed), CEED_ERROR_BACKEND, nvrtcGetErrorString(result)); \
  } while (0)

#define CeedCallNvrtc(ceed, ...)  \
  do {                            \
    int ierr_q_ = __VA_ARGS__;    \
    CeedChk_Nvrtc(ceed, ierr_q_); \
  } while (0)

#define CeedCallSystem(ceed, command, message) CeedCallBackend(CeedCallSystem_Core(ceed, command, message))

//------------------------------------------------------------------------------
// Call system command and capture stdout + stderr
//------------------------------------------------------------------------------
static int CeedCallSystem_Core(Ceed ceed, const char *command, const char *message) {
  CeedDebug(ceed, "Running command:\n$ %s", command);
  FILE *output_stream = popen((command + std::string(" 2>&1")).c_str(), "r");

  CeedCheck(output_stream != nullptr, ceed, CEED_ERROR_BACKEND, "Failed to %s\ncommand:\n$ %s", message, command);

  char        line[CEED_MAX_RESOURCE_LEN] = "";
  std::string output                      = "";

  while (fgets(line, sizeof(line), output_stream) != nullptr) {
    output += line;
  }
  CeedDebug(ceed, "output:\n%s\n", output.c_str());
  CeedCheck(pclose(output_stream) == 0, ceed, CEED_ERROR_BACKEND, "Failed to %s\ncommand:\n$ %s\nerror:\n%s", message, command, output.c_str());
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compile CUDA kernel
//------------------------------------------------------------------------------
using std::ifstream;
using std::ofstream;
using std::ostringstream;

static int CeedCompileCore_Cuda(Ceed ceed, const char *source, const bool throw_error, bool *is_compile_good, CUmodule *module,
                                const CeedInt num_defines, va_list args) {
  size_t                ptx_size;
  char                 *ptx;
  const int             num_opts            = 4;
  CeedInt               num_jit_source_dirs = 0, num_jit_defines = 0;
  const char          **opts;
  nvrtcProgram          prog;
  struct cudaDeviceProp prop;
  Ceed_Cuda            *ceed_data;

  cudaFree(0);  // Make sure a Context exists for nvrtc

  std::ostringstream code;
  bool               using_clang;

  CeedCallBackend(CeedGetIsClang(ceed, &using_clang));

  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS,
               using_clang ? "Compiling CUDA with Clang backend (with Rust QFunction support)"
                           : "Compiling CUDA with NVRTC backend (without Rust QFunction support).\nTo use the Clang backend, set the environment "
                             "variable GPU_CLANG=1");

  // Get kernel specific options, such as kernel constants
  if (num_defines > 0) {
    char *name;
    int   val;

    for (int i = 0; i < num_defines; i++) {
      name = va_arg(args, char *);
      val  = va_arg(args, int);
      code << "#define " << name << " " << val << "\n";
    }
  }

  // Standard libCEED definitions for CUDA backends
  code << "#include <ceed/jit-source/cuda/cuda-jit.h>\n\n";

  // Non-macro options
  CeedCallBackend(CeedCalloc(num_opts, &opts));
  opts[0] = "-default-device";
  CeedCallBackend(CeedGetData(ceed, &ceed_data));
  CeedCallCuda(ceed, cudaGetDeviceProperties(&prop, ceed_data->device_id));
  std::string arch_arg =
#if CUDA_VERSION >= 11010
      // NVRTC used to support only virtual architectures through the option
      // -arch, since it was only emitting PTX. It will now support actual
      // architectures as well to emit SASS.
      // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#dynamic-code-generation
      "-arch=sm_"
#else
      "-arch=compute_"
#endif
      + std::to_string(prop.major) + std::to_string(prop.minor);
  opts[1] = arch_arg.c_str();
  opts[2] = "-Dint32_t=int";
  opts[3] = "-DCEED_RUNNING_JIT_PASS=1";
  // Additional include dirs
  {
    const char **jit_source_dirs;

    CeedCallBackend(CeedGetJitSourceRoots(ceed, &num_jit_source_dirs, &jit_source_dirs));
    CeedCallBackend(CeedRealloc(num_opts + num_jit_source_dirs, &opts));
    for (CeedInt i = 0; i < num_jit_source_dirs; i++) {
      std::ostringstream include_dir_arg;

      include_dir_arg << "-I" << jit_source_dirs[i];
      CeedCallBackend(CeedStringAllocCopy(include_dir_arg.str().c_str(), (char **)&opts[num_opts + i]));
    }
    CeedCallBackend(CeedRestoreJitSourceRoots(ceed, &jit_source_dirs));
  }
  // User defines
  {
    const char **jit_defines;

    CeedCallBackend(CeedGetJitDefines(ceed, &num_jit_defines, &jit_defines));
    CeedCallBackend(CeedRealloc(num_opts + num_jit_source_dirs + num_jit_defines, &opts));
    for (CeedInt i = 0; i < num_jit_defines; i++) {
      std::ostringstream define_arg;

      define_arg << "-D" << jit_defines[i];
      CeedCallBackend(CeedStringAllocCopy(define_arg.str().c_str(), (char **)&opts[num_opts + num_jit_source_dirs + i]));
    }
    CeedCallBackend(CeedRestoreJitDefines(ceed, &jit_defines));
  }

  // Add string source argument provided in call
  code << source;

  // Compile kernel
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- ATTEMPTING TO COMPILE JIT SOURCE ----------\n");
  CeedDebug(ceed, "Source:\n%s\n", code.str().c_str());
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- END OF JIT SOURCE ----------\n");

  if (!using_clang) {
    CeedCallNvrtc(ceed, nvrtcCreateProgram(&prog, code.str().c_str(), NULL, 0, NULL, NULL));

    if (CeedDebugFlag(ceed)) {
      // LCOV_EXCL_START
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- JiT COMPILER OPTIONS ----------\n");
      for (CeedInt i = 0; i < num_opts + num_jit_source_dirs + num_jit_defines; i++) {
        CeedDebug(ceed, "Option %d: %s", i, opts[i]);
      }
      CeedDebug(ceed, "");
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- END OF JiT COMPILER OPTIONS ----------\n");
      // LCOV_EXCL_STOP
    }

    nvrtcResult result = nvrtcCompileProgram(prog, num_opts + num_jit_source_dirs + num_jit_defines, opts);

    for (CeedInt i = 0; i < num_jit_source_dirs; i++) {
      CeedCallBackend(CeedFree(&opts[num_opts + i]));
    }
    for (CeedInt i = 0; i < num_jit_defines; i++) {
      CeedCallBackend(CeedFree(&opts[num_opts + num_jit_source_dirs + i]));
    }
    CeedCallBackend(CeedFree(&opts));
    *is_compile_good = result == NVRTC_SUCCESS;
    if (!*is_compile_good) {
      char  *log;
      size_t log_size;

      CeedCallNvrtc(ceed, nvrtcGetProgramLogSize(prog, &log_size));
      CeedCallBackend(CeedMalloc(log_size, &log));
      CeedCallNvrtc(ceed, nvrtcGetProgramLog(prog, log));
      if (throw_error) {
        return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s", nvrtcGetErrorString(result), log);
      } else {
        // LCOV_EXCL_START
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
        CeedDebug(ceed, "Error: %s\nCompile log:\n%s\n", nvrtcGetErrorString(result), log);
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
        CeedCallBackend(CeedFree(&log));
        CeedCallNvrtc(ceed, nvrtcDestroyProgram(&prog));
        return CEED_ERROR_SUCCESS;
        // LCOV_EXCL_STOP
      }
    }

#if CUDA_VERSION >= 11010
    CeedCallNvrtc(ceed, nvrtcGetCUBINSize(prog, &ptx_size));
    CeedCallBackend(CeedMalloc(ptx_size, &ptx));
    CeedCallNvrtc(ceed, nvrtcGetCUBIN(prog, ptx));
#else
    CeedCallNvrtc(ceed, nvrtcGetPTXSize(prog, &ptx_size));
    CeedCallBackend(CeedMalloc(ptx_size, &ptx));
    CeedCallNvrtc(ceed, nvrtcGetPTX(prog, ptx));
#endif
    CeedCallNvrtc(ceed, nvrtcDestroyProgram(&prog));

    CeedCallCuda(ceed, cuModuleLoadData(module, ptx));
    CeedCallBackend(CeedFree(&ptx));
    return CEED_ERROR_SUCCESS;
  } else {
    srand(time(NULL));
    const int build_id = rand();

    // Create temp dir if needed
    {
      DIR *dir = opendir("temp");

      if (dir) {
        closedir(dir);
      } else {
        // In parallel multiple processes may attempt
        // Only one process needs to succeed
        mkdir("temp", 0777);
        chmod("temp", 0777);
      }
    }
    // Write code to temp file
    {
      std::string filename = std::string("temp/kernel_") + std::to_string(build_id) + std::string("_0_source.cu");
      FILE       *file     = fopen(filename.c_str(), "w");

      CeedCheck(file, ceed, CEED_ERROR_BACKEND, "Failed to create file. Write access is required for cuda-clang");
      fputs(code.str().c_str(), file);
      fclose(file);
    }

    // Get rust crate directories
    const char **rust_source_dirs     = nullptr;
    int          num_rust_source_dirs = 0;

    CeedCallBackend(CeedGetRustSourceRoots(ceed, &num_rust_source_dirs, &rust_source_dirs));

    std::string rust_dirs[10];

    if (num_rust_source_dirs > 0) {
      CeedDebug(ceed, "There are %d source dirs, including %s\n", num_rust_source_dirs, rust_source_dirs[0]);
    }

    for (CeedInt i = 0; i < num_rust_source_dirs; i++) {
      rust_dirs[i] = std::string(rust_source_dirs[i]);
    }

    CeedCallBackend(CeedRestoreRustSourceRoots(ceed, &rust_source_dirs));

    char *rust_toolchain = std::getenv("RUST_TOOLCHAIN");

    if (rust_toolchain == nullptr) {
      rust_toolchain = (char *)"nightly";
      setenv("RUST_TOOLCHAIN", "nightly", 0);
    }

    // Compile Rust crate(s) needed
    std::string command;

    for (CeedInt i = 0; i < num_rust_source_dirs; i++) {
      command = "cargo +" + std::string(rust_toolchain) + " build --release --target nvptx64-nvidia-cuda --config " + rust_dirs[i] +
                "/.cargo/config.toml --manifest-path " + rust_dirs[i] + "/Cargo.toml";
      CeedCallSystem(ceed, command.c_str(), "build Rust crate");
    }

    // Get Clang version
    bool use_llvm_version = ceed_data->use_llvm_version;
    int  llvm_version     = ceed_data->llvm_version;

    if (llvm_version == 0) {
      command = "$(find $(rustup run " + std::string(rust_toolchain) + " rustc --print sysroot) -name llvm-link) --version";
      CeedDebug(ceed, "Attempting to detect Rust LLVM version.\ncommand:\n$ %s", command.c_str());
      FILE *output_stream = popen((command + std::string(" 2>&1")).c_str(), "r");

      CeedCheck(output_stream != nullptr, ceed, CEED_ERROR_BACKEND, "Failed to detect Rust LLVM version");

      char        line[CEED_MAX_RESOURCE_LEN] = "";
      std::string output                      = "";

      while (fgets(line, sizeof(line), output_stream) != nullptr) {
        output += line;
      }
      CeedDebug(ceed, "output:\n%s", output.c_str());
      CeedCheck(pclose(output_stream) == 0, ceed, CEED_ERROR_BACKEND, "Failed to detect Rust LLVM version\ncommand:\n$ %s\nerror:\n%s",
                command.c_str(), output.c_str());

      const char *version_substring = strstr(output.c_str(), "LLVM version ");

      version_substring += 13;

      char *next_dot = strchr((char *)version_substring, '.');

      next_dot[0]             = '\0';
      ceed_data->llvm_version = llvm_version = std::stoi(version_substring);
      CeedDebug(ceed, "Rust LLVM version number: %d\n", llvm_version);

      command                     = std::string("clang++-") + std::to_string(llvm_version);
      output_stream               = popen((command + std::string(" 2>&1")).c_str(), "r");
      ceed_data->use_llvm_version = use_llvm_version = pclose(output_stream) == 0;
    }

    // Compile wrapper kernel
    command = "clang++" + (use_llvm_version ? (std::string("-") + std::to_string(llvm_version)) : "") + " -flto=thin --cuda-gpu-arch=sm_" +
              std::to_string(prop.major) + std::to_string(prop.minor) + " --cuda-device-only -emit-llvm -S temp/kernel_" + std::to_string(build_id) +
              "_0_source.cu -o temp/kernel_" + std::to_string(build_id) + "_1_wrapped.ll ";
    command += opts[4];
    CeedCallSystem(ceed, command.c_str(), "JiT kernel source");
    CeedCallSystem(ceed, ("chmod 0777 temp/kernel_" + std::to_string(build_id) + "_1_wrapped.ll").c_str(), "update JiT file permissions");

    // Find Rust's llvm-link tool and runs it
    command = "$(find $(rustup run " + std::string(rust_toolchain) + " rustc --print sysroot) -name llvm-link) temp/kernel_" +
              std::to_string(build_id) +
              "_1_wrapped.ll --ignore-non-bitcode --internalize --only-needed -S -o "
              "temp/kernel_" +
              std::to_string(build_id) + "_2_linked.ll ";

    // Searches for .a files in rust directoy
    // Note: this is necessary because Rust crate names may not match the folder they are in
    for (CeedInt i = 0; i < num_rust_source_dirs; i++) {
      std::string dir = rust_dirs[i] + "/target/nvptx64-nvidia-cuda/release";
      DIR        *dp  = opendir(dir.c_str());

      CeedCheck(dp != nullptr, ceed, CEED_ERROR_BACKEND, "Could not open directory: %s", dir.c_str());
      struct dirent *entry;

      // Find files ending in .a
      while ((entry = readdir(dp)) != nullptr) {
        std::string filename(entry->d_name);

        if (filename.size() >= 2 && filename.substr(filename.size() - 2) == ".a") {
          command += dir + "/" + filename + " ";
        }
      }
      closedir(dp);
      // TODO: when libCEED switches to c++17, switch to std::filesystem for the loop above
    }

    // Link, optimize, and compile final CUDA kernel
    CeedCallSystem(ceed, command.c_str(), "link C and Rust source");
    CeedCallSystem(
        ceed,
        ("$(find $(rustup run " + std::string(rust_toolchain) + " rustc --print sysroot) -name opt) --passes internalize,inline temp/kernel_" +
         std::to_string(build_id) + "_2_linked.ll -o temp/kernel_" + std::to_string(build_id) + "_3_opt.bc")
            .c_str(),
        "optimize linked C and Rust source");
    CeedCallSystem(ceed, ("chmod 0777 temp/kernel_" + std::to_string(build_id) + "_2_linked.ll").c_str(), "update JiT file permissions");
    CeedCallSystem(ceed,
                   ("$(find $(rustup run " + std::string(rust_toolchain) + " rustc --print sysroot) -name llc) -O3 -mcpu=sm_" +
                    std::to_string(prop.major) + std::to_string(prop.minor) + " temp/kernel_" + std::to_string(build_id) +
                    "_3_opt.bc -o temp/kernel_" + std::to_string(build_id) + "_4_final.ptx")
                       .c_str(),
                   "compile final CUDA kernel");
    CeedCallSystem(ceed, ("chmod 0777 temp/kernel_" + std::to_string(build_id) + "_4_final.ptx").c_str(), "update JiT file permissions");

    ifstream      ptxfile("temp/kernel_" + std::to_string(build_id) + "_4_final.ptx");
    ostringstream sstr;

    sstr << ptxfile.rdbuf();

    auto ptx_data = sstr.str();
    ptx_size      = ptx_data.length();

    int result = cuModuleLoadData(module, ptx_data.c_str());

    *is_compile_good = result == 0;
    if (!*is_compile_good) {
      if (throw_error) {
        return CeedError(ceed, CEED_ERROR_BACKEND, "Failed to load module data");
      } else {
        // LCOV_EXCL_START
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
        CeedDebug(ceed, "Error: Failed to load module data");
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
        return CEED_ERROR_SUCCESS;
        // LCOV_EXCL_STOP
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

int CeedCompile_Cuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...) {
  bool    is_compile_good = true;
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cuda(ceed, source, true, &is_compile_good, module, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedTryCompile_Cuda(Ceed ceed, const char *source, bool *is_compile_good, CUmodule *module, const CeedInt num_defines, ...) {
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cuda(ceed, source, false, is_compile_good, module, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get CUDA kernel
//------------------------------------------------------------------------------
int CeedGetKernel_Cuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel) {
  CeedCallCuda(ceed, cuModuleGetFunction(kernel, module, name));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel with block size selected automatically based on the kernel
//     (which may use enough registers to require a smaller block size than the
//      hardware is capable)
//------------------------------------------------------------------------------
int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points, void **args) {
  int min_grid_size, max_block_size;

  CeedCallCuda(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_block_size, kernel, NULL, 0, 0x10000));
  CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(points, max_block_size), max_block_size, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel
//------------------------------------------------------------------------------
int CeedRunKernel_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size, void **args) {
  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, NULL, grid_size, block_size, 1, 1, 0, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension
//------------------------------------------------------------------------------
int CeedRunKernelDim_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y, const int block_size_z,
                          void **args) {
  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, NULL, grid_size, block_size_x, block_size_y, block_size_z, 0, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Run CUDA kernel for spatial dimension with shared memory
//------------------------------------------------------------------------------
static int CeedRunKernelDimSharedCore_Cuda(Ceed ceed, CUfunction kernel, CUstream stream, const int grid_size, const int block_size_x,
                                           const int block_size_y, const int block_size_z, const int shared_mem_size, const bool throw_error,
                                           bool *is_good_run, void **args) {
#if CUDA_VERSION >= 9000
  cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_mem_size);
#endif
  CUresult result = cuLaunchKernel(kernel, grid_size, 1, 1, block_size_x, block_size_y, block_size_z, shared_mem_size, stream, args, NULL);

  if (result == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
    int max_threads_per_block, shared_size_bytes, num_regs;

    cuFuncGetAttribute(&max_threads_per_block, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);
    cuFuncGetAttribute(&shared_size_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
    cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
    if (throw_error) {
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d",
                       max_threads_per_block, block_size_x, block_size_y, block_size_z, shared_size_bytes, num_regs);
    } else {
      // LCOV_EXCL_START
      CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- LAUNCH ERROR DETECTED ----------\n");
      CeedDebug(ceed, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d\n",
                max_threads_per_block, block_size_x, block_size_y, block_size_z, shared_size_bytes, num_regs);
      CeedDebug256(ceed, CEED_DEBUG_COLOR_WARNING, "---------- BACKEND MAY FALLBACK ----------\n");
      // LCOV_EXCL_STOP
    }
    *is_good_run = false;
  } else CeedChk_Cu(ceed, result);
  return CEED_ERROR_SUCCESS;
}

int CeedRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, CUstream stream, const int grid_size, const int block_size_x, const int block_size_y,
                                const int block_size_z, const int shared_mem_size, void **args) {
  bool is_good_run = true;

  CeedCallBackend(CeedRunKernelDimSharedCore_Cuda(ceed, kernel, stream, grid_size, block_size_x, block_size_y, block_size_z, shared_mem_size, true,
                                                  &is_good_run, args));
  return CEED_ERROR_SUCCESS;
}

int CeedTryRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, CUstream stream, const int grid_size, const int block_size_x, const int block_size_y,
                                   const int block_size_z, const int shared_mem_size, bool *is_good_run, void **args) {
  CeedCallBackend(CeedRunKernelDimSharedCore_Cuda(ceed, kernel, stream, grid_size, block_size_x, block_size_y, block_size_z, shared_mem_size, false,
                                                  is_good_run, args));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
