// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-compile.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <ceed/gen-system.hpp>
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
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "ceed-cuda-common.h"

const char *CeedCudaDir = CEED_CUDA_DIR;

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

//------------------------------------------------------------------------------
// Build array of JIT flags
//------------------------------------------------------------------------------
static inline int CeedJitGetOpts_Cuda(Ceed ceed, const char ***opts, int *num_opts) {
  int opts_count = 4;

  // Standard options
  CeedCallBackend(CeedCalloc(opts_count, opts));
  CeedCallBackend(CeedStringAllocCopy("-default-device", (char **)&(*opts)[0]));
  {
    Ceed_Cuda            *ceed_data;
    struct cudaDeviceProp prop;

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

    CeedCallBackend(CeedStringAllocCopy(arch_arg.c_str(), (char **)&(*opts)[1]));
  }
  CeedCallBackend(CeedStringAllocCopy("-Dint32_t=int", (char **)&(*opts)[2]));
  CeedCallBackend(CeedStringAllocCopy("-DCEED_RUNNING_JIT_PASS=1", (char **)&(*opts)[3]));

  // Additional include dirs
  {
    const char **jit_source_dirs;
    CeedInt      num_jit_source_dirs;

    CeedCallBackend(CeedGetJitSourceRoots(ceed, &num_jit_source_dirs, &jit_source_dirs));
    CeedCallBackend(CeedRealloc(opts_count + num_jit_source_dirs, opts));
    for (CeedInt i = 0; i < num_jit_source_dirs; i++) {
      std::ostringstream include_dir_arg;

      include_dir_arg << "-I" << jit_source_dirs[i];
      CeedCallBackend(CeedStringAllocCopy(include_dir_arg.str().c_str(), (char **)&(*opts)[opts_count + i]));
    }
    CeedCallBackend(CeedRestoreJitSourceRoots(ceed, &jit_source_dirs));
    opts_count += num_jit_source_dirs;
  }

  // User defines
  {
    const char **jit_defines;
    CeedInt      num_jit_defines;

    CeedCallBackend(CeedGetJitDefines(ceed, &num_jit_defines, &jit_defines));
    CeedCallBackend(CeedRealloc(opts_count + num_jit_defines, opts));
    for (CeedInt i = 0; i < num_jit_defines; i++) {
      std::ostringstream define_arg;

      define_arg << "-D" << jit_defines[i];
      CeedCallBackend(CeedStringAllocCopy(define_arg.str().c_str(), (char **)&(*opts)[opts_count + i]));
    }
    CeedCallBackend(CeedRestoreJitDefines(ceed, &jit_defines));
    opts_count += num_jit_defines;
  }
  *num_opts = opts_count;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Compile CUDA kernel
//------------------------------------------------------------------------------
using std::ifstream;
using std::ofstream;
using std::ostringstream;

static int CeedCompileCore_Cuda(Ceed ceed, const char *source, const char *name, const bool throw_error, bool *is_compile_good, CUmodule *module,
                                const CeedInt num_defines, va_list args) {
  bool               using_clang;
  size_t             ptx_size;
  char              *ptx;
  nvrtcProgram       prog;
  std::ostringstream code;

  // Make sure a Context exists for nvrtc
  cudaFree(0);

  CeedCallBackend(CeedGetCudaUseClang(ceed, &using_clang));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS,
               using_clang ? "Compiling CUDA with Clang backend (with Rust QFunction support)"
                           : "Compiling CUDA with NVRTC backend (without Rust QFunction support)."
                             "\nTo use the Clang backend, set the environment variable CEED_USE_CLANG_CUDA=1"
                             " or CEED_CLANG_CUDA_CXX=my_clang++");

  // Get kernel specific options, such as kernel constants
  if (num_defines > 0) {
    char *name;
    int   val;

    for (int i = 0; i < num_defines; i++) {
      name = va_arg(args, char *);
      val  = va_arg(args, int);
      code << "#define " << name << " " << val << "\n\n";
    }
  }

  // Standard libCEED definitions for CUDA backends
  code << "#include <ceed/jit-source/cuda/cuda-jit.h>\n\n";

  // Add string source argument provided in call
  code << source;

  // Compile kernel
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- ATTEMPTING TO COMPILE JIT SOURCE ----------\n");
  CeedDebug(ceed, "Name:\n  %s\n", name);
  CeedDebug(ceed, "Source:\n%s\n", code.str().c_str());
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- END OF JIT SOURCE ----------\n");

  // Write to disk in debug mode
  if (CeedDebugFlag(ceed)) {
    // LCOV_EXCL_START
    // Create filename with path and 'function_' prefix with uuid
    std::random_device         r;
    std::default_random_engine gen(r());
    // Place lower bound for uniformity of ids
    std::uniform_int_distribution<CeedInt> dist(1000000000);
    const CeedInt                          build_id = dist(gen);
    std::string                            filename_cpp;

    {
      const char *dir;

      CeedCallBackend(CeedGetCacheDir(ceed, &dir));
      filename_cpp = std::string(dir) + "/function_" + std::to_string(build_id) + "_" + name + ".cu";
      CeedCallBackend(CeedRestoreCacheDir(ceed, &dir));
    }

    // Write code to temp file
    FILE *file = fopen(filename_cpp.c_str(), "w");

    CeedCheck(file, ceed, CEED_ERROR_BACKEND, "Failed to create file. Write access is required for cpu-jit");
    fputs(code.str().c_str(), file);
    fclose(file);
    // LCOV_EXCL_STOP
  }

  if (!using_clang) {
    const char **opts;
    int          num_opts;

    // Get compile options
    CeedCallBackend(CeedJitGetOpts_Cuda(ceed, &opts, &num_opts));
    CeedCallNvrtc(ceed, nvrtcCreateProgram(&prog, code.str().c_str(), NULL, 0, NULL, NULL));

    if (CeedDebugFlag(ceed)) {
      // LCOV_EXCL_START
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- JiT COMPILER OPTIONS ----------\n");
      for (CeedInt i = 0; i < num_opts; i++) CeedDebug(ceed, "Option %d: %s", i, opts[i]);
      CeedDebug(ceed, "");
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- END OF JiT COMPILER OPTIONS ----------\n");
      // LCOV_EXCL_STOP
    }
    nvrtcResult result = nvrtcCompileProgram(prog, num_opts, opts);

    for (CeedInt i = 0; i < num_opts; i++) CeedCallBackend(CeedFree(&opts[i]));
    CeedCallBackend(CeedFree(&opts));

    *is_compile_good = result == NVRTC_SUCCESS;
    if (!*is_compile_good) {
      // LCOV_EXCL_START
      char  *log;
      size_t log_size;

      CeedCallNvrtc(ceed, nvrtcGetProgramLogSize(prog, &log_size));
      CeedCallBackend(CeedMalloc(log_size, &log));
      CeedCallNvrtc(ceed, nvrtcGetProgramLog(prog, log));
      if (throw_error) {
        return CeedError(ceed, CEED_ERROR_BACKEND, "%s\n%s", nvrtcGetErrorString(result), log);
      } else {
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
        CeedDebug(ceed, "Error: %s\nCompile log:\n%s\n", nvrtcGetErrorString(result), log);
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
        CeedCallBackend(CeedFree(&log));
        CeedCallNvrtc(ceed, nvrtcDestroyProgram(&prog));
        return CEED_ERROR_SUCCESS;
      }
      // LCOV_EXCL_STOP
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
    std::random_device         r;
    std::default_random_engine gen(r());
    // Place lower bound for uniformity of ids
    std::uniform_int_distribution<CeedInt> dist(1000000000);
    const CeedInt                          build_id = dist(gen);
    struct cudaDeviceProp                  prop;
    std::string                            cache_dir, filename_ptx;
    bool                                   ptx_file_exists;

    {
      const char *dir;

      CeedCallBackend(CeedGetCacheDir(ceed, &dir));
      cache_dir = std::string(dir) + "/";
      CeedCallBackend(CeedRestoreCacheDir(ceed, &dir));
    }

    // Get rust crate directories
    const char              *rust_toolchain;
    const char             **rust_source_dirs     = nullptr;
    int                      num_rust_source_dirs = 0;
    std::vector<std::string> rust_dirs;
    std::string              toolchain;
    CeedCallSystemResult     result;

    CeedCallBackend(CeedGetRustSourceRoots(ceed, &num_rust_source_dirs, &rust_source_dirs));
    for (CeedInt i = 0; i < num_rust_source_dirs; i++) {
      rust_dirs.push_back(rust_source_dirs[i]);
    }
    CeedCallBackend(CeedRestoreRustSourceRoots(ceed, &rust_source_dirs));

    CeedCallBackend(CeedGetCudaRustupToolchain(ceed, &rust_toolchain));

    // Compile Rust crate(s) needed
    std::string command;

    for (CeedInt i = 0; i < num_rust_source_dirs; i++) {
      command = "cargo +" + std::string(rust_toolchain) + " build --release --target nvptx64-nvidia-cuda --config " + rust_dirs[i] +
                "/.cargo/config.toml --manifest-path " + rust_dirs[i] + "/Cargo.toml";
      CeedCallSystem(ceed, command, "build Rust crate", result);
    }

    // Get Clang version
    Ceed_Cuda *ceed_data;

    CeedCallBackend(CeedGetData(ceed, &ceed_data));
    char *llvm_cxx = ceed_data->llvm_cxx;

    // First check for user LLVM version
    if (!llvm_cxx) {
      const char *user_cxx;

      CeedCallBackend(CeedGetCudaClangCxx(ceed, &user_cxx));
      CeedDebug(ceed, "Attempting to detect user specified LLVM compiler\nUser LLVM compiler: %s\n", user_cxx);

      // Check if valid Clang
      result.is_success = false;
      if (user_cxx && *user_cxx != '\0') {
        CeedDebug(ceed, "Checking user LLVM compiler...");
        CeedCall(CeedCallSystemUnchecked(ceed, std::string(user_cxx) + " --version", "checking user LLVM compiler", result));
      }
      if (result.is_success) {
        CeedDebug(ceed, "User specified LLVM compiler is valid\n");
        CeedCallBackend(CeedStringAllocCopy(user_cxx, &ceed_data->llvm_cxx));
        llvm_cxx = ceed_data->llvm_cxx;
      } else {
        CeedDebug(ceed, "Could not invoke user specified LLVM compiler\n");
      }
    }
    // Next query Rust for LLVM version
    if (!llvm_cxx) {
      command = "$(find $(rustup run " + std::string(rust_toolchain) + " rustc --print sysroot) -name llvm-link) --version";
      CeedCall(CeedCallSystemUnchecked(ceed, command, "detect Rust LLVM version", result));

      if (result.is_success) {
        CeedDebug(ceed, "output:\n%s", result.output.c_str());

        auto version_substring_start = result.output.find("LLVM version ");
        if (version_substring_start != std::string::npos) version_substring_start += 13;
        auto version_substring_end = result.output.find(".", version_substring_start);

        if (version_substring_end > version_substring_start && version_substring_start != std::string::npos &&
            version_substring_end != std::string::npos) {
          auto llvm_version_str = result.output.substr(version_substring_start, version_substring_end - version_substring_start);
          CeedDebug(ceed, "Detected Rust LLVM version: %s", llvm_version_str.c_str());
          CeedInt llvm_version = std::stoi(llvm_version_str);
          CeedDebug(ceed, "Detected Rust LLVM version: %d", llvm_version);

          // Check if valid Clang
          std::string rust_cxx = std::string("clang++-") + std::to_string(llvm_version);

          CeedDebug(ceed, "Checking Rust LLVM compiler...");
          CeedCall(CeedCallSystemUnchecked(ceed, std::string(rust_cxx) + " --version", "checking Rust LLVM compiler", result));

          if (result.is_success) {
            CeedDebug(ceed, "Detected Rust LLVM compiler: %s\n", rust_cxx.c_str());
            CeedCall(CeedStringAllocCopy(rust_cxx.c_str(), &ceed_data->llvm_cxx));
            llvm_cxx = ceed_data->llvm_cxx;
          }
        }
        if (!llvm_cxx) CeedDebug(ceed, "Could not invoke detected Rust LLVM compiler\n");
      }
    }
    // Default to clang++
    if (!llvm_cxx) {
      CeedDebug(ceed, "Default LLVM compiler: clang++\n");
      CeedDebug(ceed, "Checking default LLVM compiler...");
      CeedCallSystemUnchecked(ceed, "clang++ --version", "checking default LLVM compiler", result);
      if (result.is_success) {
        CeedCall(CeedStringAllocCopy("clang++", &ceed_data->llvm_cxx));
        llvm_cxx = ceed_data->llvm_cxx;
      }
    }

    if (!llvm_cxx) {
      // LCOV_EXCL_START
      *is_compile_good = false;
      if (throw_error) {
        return CeedError(ceed, CEED_ERROR_BACKEND, "Failed to find LLVM compiler");
      } else {
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
        CeedDebug(ceed, "Error: Failed to find LLVM compiler");
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
        return CEED_ERROR_SUCCESS;
      }
      // LCOV_EXCL_STOP
    }

    std::string code_str = code.str();
    std::string includes;

    {
      CeedInt      num_source_roots;
      const char **source_roots;

      CeedCall(CeedGetJitSourceRoots(ceed, &num_source_roots, &source_roots));
      for (CeedInt i = 0; i < num_source_roots; i++) includes += " -I" + std::string(source_roots[i]);
      CeedCall(CeedRestoreJitSourceRoots(ceed, &source_roots));
    }

    CeedCallCuda(ceed, cudaGetDeviceProperties(&prop, ceed_data->device_id));

    // Preprocess & check if identical file has been compiled
    {
      // -E: preprocess only
      // -P: exclude line info (needed to support different filenames)
      CeedCallSystemResult result;
      std::string          command = std::string(llvm_cxx) + " --cuda-path=" + std::string(CeedCudaDir) + " -flto=thin --cuda-gpu-arch=sm_" +
                                     std::to_string(prop.major) + std::to_string(prop.minor) + includes + " --cuda-device-only -x cu -E -P - -o -";

      CeedCallSystem(ceed, command, "JiT preprocess function source into memory", code_str, result);

      std::size_t cpp_hash = std::hash<std::string>{}(result.output);
      filename_ptx         = cache_dir + "function_" + std::to_string(cpp_hash) + "_" + name + ".ptx";

      // Fast way to check if a file exists
      struct stat buffer;
      ptx_file_exists = (stat(filename_ptx.c_str(), &buffer) == 0);
    }

    // Compile wrapper kernel
    if (!ptx_file_exists) {
      // Compile wrapper kernel
      command = std::string(llvm_cxx) + " --cuda-path=" + std::string(CeedCudaDir) + " -flto=thin --cuda-gpu-arch=sm_" + std::to_string(prop.major) +
                std::to_string(prop.minor) + includes + " --cuda-device-only -emit-llvm -S -x cu - -o -";
      CeedCall(CeedCallSystem(ceed, command, "JiT kernel source", code_str, result));

      std::string tmp_ptx_filename = cache_dir + ".tmp_function_" + std::to_string(build_id) + "_" + name + +".cubin";

      // Find Rust's llvm-link tool and run it
      command = "$(find $(rustup run " + std::string(rust_toolchain) +
                " rustc --print sysroot) -name llvm-link) - --ignore-non-bitcode --internalize --only-needed -S ";
      // Searches for .a files in Rust directory
      // Note: Rust crate names may not match the folder they are in
      // TODO: If libCEED switches to c++17, use std::filesystem here
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
      }
      command += "-o -";

      // Link, optimize, and compile final CUDA kernel
      CeedCallSystem(ceed, command, "link C and Rust source", result.output, result);
      command = "$(find $(rustup run " + std::string(rust_toolchain) + " rustc --print sysroot) -name opt) --passes internalize,inline - -o - ";
      CeedCallSystem(ceed, command, "optimize linked C and Rust source", result.output, result);

      // As of now, the .ptx doesn't exist, but another process might be compiling simultaneously
      // So, compile to temporary file and use link() (guaranteed atomic by POSIX) to try to move
      command = "$(find $(rustup run " + std::string(rust_toolchain) + " rustc --print sysroot) -name llc) -O3 -mcpu=sm_" +
                std::to_string(prop.major) + std::to_string(prop.minor) + " - -o " + tmp_ptx_filename;
      CeedCallSystem(ceed, command, "compile final CUDA kernel", result.output, result);
      CeedCallSystem(ceed, "chmod 0777 " + tmp_ptx_filename, "update JiT file permissions", result);

      // Atomicly try to move to final location
      if (link(tmp_ptx_filename.c_str(), filename_ptx.c_str()) < 0) {
        // EEXIST means another process beat us to it, so succeed silently
        CeedCheck(errno == EEXIST, ceed, CEED_ERROR_BACKEND, "Failed to write '%s' to disk: %s", filename_ptx.c_str(), strerror(errno));
        errno = 0;
      }

      // Remove temporary file
      {
        int err = errno;

        unlink(tmp_ptx_filename.c_str());
        errno = err;
      }
    }

    // Load module from final PTX
    ifstream    ptxfile(filename_ptx);
    std::string buf(std::istreambuf_iterator<char>(ptxfile), {});
    int         load_result = cuModuleLoadData(module, buf.c_str());

    *is_compile_good = load_result == 0;
    if (!*is_compile_good) {
      // LCOV_EXCL_START
      if (throw_error) {
        return CeedError(ceed, CEED_ERROR_BACKEND, "Failed to load module data");
      } else {
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
        CeedDebug(ceed, "Error: Failed to load module data");
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
        return CEED_ERROR_SUCCESS;
      }
      // LCOV_EXCL_STOP
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <typename ArrayT>
struct CeedArrayView {
  const ArrayT *array;
  CeedInt       size;

  CeedArrayView(const ArrayT *array_, CeedInt size_) : array(array_), size(size_) {}
};

template <typename OStream, typename ArrayT>
OStream &operator<<(OStream &ostream, const CeedArrayView<ArrayT> &view) {
  ostream << "{";
  for (CeedInt i = 0; i < view.size; i++) ostream << std::setprecision(17) << view.array[i] << (i == view.size - 1 ? "}" : ", ");
  return ostream;
}

int CeedBuildArrayConstantSize_Cuda(Ceed ceed, const char *name, CeedInt length, const CeedSize *array, char **line) {
  std::ostringstream code;

  code << "constexpr CeedSize " << name << "[" << length << "] = " << CeedArrayView<CeedSize>(array, length) << ";";
  CeedCallBackend(CeedStringAllocCopy(code.str().c_str(), line));
  return CEED_ERROR_SUCCESS;
}

int CeedCompile_Cuda(Ceed ceed, const char *source, const char *name, CUmodule *module, const CeedInt num_defines, ...) {
  bool    is_compile_good = true;
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cuda(ceed, source, name, true, &is_compile_good, module, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedTryCompile_Cuda(Ceed ceed, const char *source, const char *name, bool *is_compile_good, CUmodule *module, const CeedInt num_defines, ...) {
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cuda(ceed, source, name, false, is_compile_good, module, num_defines, args);

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
    // LCOV_EXCL_START
    int max_threads_per_block, shared_size_bytes, num_regs;

    cuFuncGetAttribute(&max_threads_per_block, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel);
    cuFuncGetAttribute(&shared_size_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
    cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel);
    if (throw_error) {
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d",
                       max_threads_per_block, block_size_x, block_size_y, block_size_z, shared_size_bytes, num_regs);
    } else {
      CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- LAUNCH ERROR DETECTED ----------\n");
      CeedDebug(ceed, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d\n",
                max_threads_per_block, block_size_x, block_size_y, block_size_z, shared_size_bytes, num_regs);
      CeedDebug256(ceed, CEED_DEBUG_COLOR_WARNING, "---------- BACKEND MAY FALLBACK ----------\n");
    }
    // LCOV_EXCL_STOP
    *is_good_run = false;
  } else {
    CeedChk_Cu(ceed, result);
  }
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
