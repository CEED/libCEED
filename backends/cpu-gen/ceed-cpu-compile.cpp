// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cpu-compile.h"
#include "ceed-cpu-gen.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <ceed/gen-system.hpp>
#include <dlfcn.h>
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
#include <unistd.h>

#define CEED_QUOTE(name) #name
#define CEED_STRINGIFY(macro) CEED_QUOTE(macro)
const char *CeedJitCxxDefault = CEED_STRINGIFY(CEED_CPU_JIT_CXX);
#undef CEED_QUOTE
#undef CEED_STRINGIFY

//------------------------------------------------------------------------------
// Build array of JIT flags
//------------------------------------------------------------------------------
static inline int CeedJitGetOpts_Cpu(Ceed ceed, const char ***opts, int *num_opts) {
  int opts_count = 1;

  // Standard options
  CeedCallBackend(CeedCalloc(opts_count, opts));
  {
    const char *jit_opt;

    CeedCallBackend(CeedGetCpuJitOpt(ceed, &jit_opt));
    CeedCallBackend(CeedStringAllocCopy(jit_opt, (char **)&(*opts)[0]));
  }

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
// Compile CPU function
//------------------------------------------------------------------------------
using std::ifstream;
using std::ofstream;
using std::ostringstream;

static inline int CeedCompileCore_Cpu(Ceed ceed, const char *source, const char *name, const bool throw_error, bool *is_compile_good, void **handle,
                                      const CeedInt num_defines, va_list args) {
  const char       **opts;
  int                num_opts;
  std::ostringstream code;
  Ceed_Cpu_Gen      *ceed_data;

  // Get CXX version
  CeedCallBackend(CeedGetData(ceed, &ceed_data));
  char *cxx = ceed_data->cxx;

  // First check for user JiT compiler
  if (!cxx) {
    CeedCallSystemResult result;
    const char          *user_cxx;

    CeedCall(CeedGetCpuJitCxx(ceed, &user_cxx));
    CeedDebug(ceed, "Attempting to detect user specified JiT compiler\nUser JiT compiler: %s\n", user_cxx);

    // Check if valid compiler
    if (user_cxx) {
      std::string command = std::string(user_cxx) + " --version 2>&1", output;

      CeedDebug(ceed, "Checking user JiT compiler...");
      CeedCallSystemUnchecked(ceed, command, "checking user JiT compiler", result);
      if (result.is_success) {
        CeedDebug(ceed, "User JiT compiler is valid\n");
        CeedCall(CeedStringAllocCopy(user_cxx, &ceed_data->cxx));
        cxx = ceed_data->cxx;
      } else {
        CeedDebug(ceed, "Could not invoke user specified JiT compiler\n");
      }
    }
  }
  // Fallback to CXX compiler used for building libCEED
  if (!cxx) {
    CeedCallSystemResult result;
    std::string          command = std::string(CeedJitCxxDefault) + " --version 2>&1";

    CeedDebug(ceed, "Default JiT compiler: %s\n", CeedJitCxxDefault);
    CeedDebug(ceed, "Checking default JiT compiler...");
    CeedCallSystemUnchecked(ceed, command, "checking default JiT compiler", result);
    if (result.is_success) {
      CeedDebug(ceed, "Default JiT compiler is valid\n");
      CeedCall(CeedStringAllocCopy(CeedJitCxxDefault, &ceed_data->cxx));
      cxx = ceed_data->cxx;
    } else {
      CeedDebug(ceed, "Could not invoke default JiT compiler\n");
    }
  }
  // Fail early if compiler doesn't work
  // LCOV_EXCL_START
  if (!cxx) {
    *is_compile_good = false;
    return CEED_ERROR_SUCCESS;
  }
  // LCOV_EXCL_STOP

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
  code << "#include <ceed/jit-source/cpu-gen/cpu-jit.h>\n\n";

  // Add string source argument provided in call
  code << source;

  // Get compile options
  CeedCallBackend(CeedJitGetOpts_Cpu(ceed, &opts, &num_opts));

  // Encode options into source
  code << "\n\n";
  code << "const static char *__ceed_compile_options[] = {\n";
  code << "  \"" << cxx << "\",\n";
  code << "  \"-shared\",\n";
  code << "  \"-fPIC\",\n";
  code << "  \"-rdynamic\",\n";
  for (CeedInt i = 0; i < num_opts; i++) {
    code << "  \"" << opts[i] << "\",\n";
  }
  code << "};\n";

  // Compile kernel
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- ATTEMPTING TO COMPILE JIT SOURCE ----------\n");
  CeedDebug(ceed, "Name:\n  %s\n", name);
  CeedDebug(ceed, "Source:\n%s\n", code.str().c_str());
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- END OF JIT SOURCE ----------\n");

  {
    // Create filename with path and 'function_' prefix with uuid
    std::random_device         r;
    std::default_random_engine gen(r());
    // Place lower bound for uniformity of ids
    std::uniform_int_distribution<CeedInt> dist(1000000000);
    const CeedInt                          build_id = dist(gen);
    std::string                            cache_dir, filename_base, filename_cpp, filename_so;

    {
      const char *dir;

      CeedCallBackend(CeedGetCacheDir(ceed, &dir));
      cache_dir = std::string(dir) + "/";
      CeedCallBackend(CeedRestoreCacheDir(ceed, &dir));
    }

    filename_cpp = cache_dir + std::string("function_") + std::to_string(build_id) + "_" + name + ".cpp";

    const std::string code_str = code.str();
    std::size_t       cpp_hash;
    bool              so_file_exists;

    // Write code to temp file
    if (CeedDebugFlag(ceed)) {
      // LCOV_EXCL_START
      FILE *file = fopen(filename_cpp.c_str(), "w");

      CeedCheck(file, ceed, CEED_ERROR_BACKEND, "Failed to create file. Write access is required for cpu-jit");
      fputs(code.str().c_str(), file);
      fclose(file);
      // LCOV_EXCL_STOP
    }

    // Preprocess & check if identical file has been compiled
    {
      // -E: preprocess only
      // -P: exclude line info (needed to support different filenames)
      std::string          command = std::string(cxx);
      CeedCallSystemResult result;

      for (CeedInt i = 0; i < num_opts; i++) command += std::string(" ") + opts[i];
      command += " -x c++ -E -P - ";
      CeedCallSystem(ceed, command, "JiT preprocess function source into memory", code_str, result);

      cpp_hash    = std::hash<std::string>{}(result.output);
      filename_so = cache_dir + "function_" + std::to_string(cpp_hash) + "_" + name + ".so";

      // Fast way to check if a file exists
      struct stat buffer;
      so_file_exists = (stat((filename_so).c_str(), &buffer) == 0);
    }

    // Compile wrapper kernel
    if (!so_file_exists) {
      std::string          command         = std::string(cxx) + " -shared -fPIC -rdynamic";
      std::string          tmp_so_filename = cache_dir + ".tmp_function_" + std::to_string(build_id) + "_" + name + +".so";
      CeedCallSystemResult result;

      // As of now, the .so doesn't exist, but another process might be compiling simultaneously
      // So, compile to temporary file and use link() (guaranteed atomic by POSIX) to try to move
      for (CeedInt i = 0; i < num_opts; i++) command += std::string(" ") + opts[i];
      command += " -x c++ -o " + tmp_so_filename + " - ";
      CeedCallSystem(ceed, command, "JiT compile function source to disk", code_str, result);
      CeedCallSystem(ceed, std::string("chmod 0777 ") + tmp_so_filename, "update JiT file permissions", result);

      // Atomicly try to move to final location
      if (link(tmp_so_filename.c_str(), filename_so.c_str()) < 0) {
        // EEXIST means another process beat us to it, so succeed silently
        CeedCheck(errno == EEXIST, ceed, CEED_ERROR_BACKEND, "Failed to write '%s' to disk: %s", filename_so.c_str(), strerror(errno));
        errno = 0;
      }

      // Remove temporary file
      {
        int err = errno;

        unlink(tmp_so_filename.c_str());
        errno = err;
      }
    }

    // Load function from object file
    CeedDebug(ceed, (std::string("Loading object file: ") + filename_so).c_str());
    *handle = dlopen((filename_so).c_str(), RTLD_NOW | RTLD_LOCAL);
    if (*handle == NULL) CeedDebug(ceed, "Error loading object file: %s", dlerror());
    *is_compile_good = *handle != NULL;

    // Check load
    if (*is_compile_good) {
      void *function;

      CeedDebug(ceed, (std::string("Loading function: ") + name).c_str());
      function = (void *)dlsym(*handle, name);
      if (function == NULL) CeedDebug(ceed, "Error loading function: %s", dlerror());
      *is_compile_good = function != NULL;
    }

    for (CeedInt i = 0; i < num_opts; i++) {
      CeedCall(CeedFree(&opts[i]));
    }
    CeedCall(CeedFree(&opts));
    if (!*is_compile_good) {
      // LCOV_EXCL_START
      if (throw_error) {
        return CeedError(ceed, CEED_ERROR_BACKEND, "Failed to load function from object file");
      } else {
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- COMPILE ERROR DETECTED ----------\n");
        CeedDebug(ceed, "Error: Failed to load function from object file\n");
        CeedDebug256(ceed, CEED_DEBUG_COLOR_ERROR, "---------- BACKEND MAY FALLBACK ----------\n");
        return CEED_ERROR_SUCCESS;
      }
      // LCOV_EXCL_STOP
    }
  }
  return CEED_ERROR_SUCCESS;
}

int CeedCompile_Cpu(Ceed ceed, const char *source, const char *name, void **handle, const CeedInt num_defines, ...) {
  bool    is_compile_good = true;
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cpu(ceed, source, name, true, &is_compile_good, handle, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedTryCompile_Cpu(Ceed ceed, const char *source, const char *name, bool *is_compile_good, void **handle, const CeedInt num_defines, ...) {
  va_list args;

  va_start(args, num_defines);
  const CeedInt ierr = CeedCompileCore_Cpu(ceed, source, name, false, is_compile_good, handle, num_defines, args);

  va_end(args);
  CeedCallBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
