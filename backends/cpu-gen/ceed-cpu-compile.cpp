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
#include <dirent.h>
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

#define CeedCallSystem(ceed, command, message) CeedCallBackend(CeedCallSystem_Core(ceed, command, message, true, NULL))
#define CeedCallSystem_Unchecked(ceed, command, message, is_success) CeedCallBackend(CeedCallSystem_Core(ceed, command, message, false, is_success))

//------------------------------------------------------------------------------
// Call system command and capture stdout + stderr
//------------------------------------------------------------------------------
static inline int CeedCallSystem_Core(Ceed ceed, const char *command, const char *message, bool err_on_fail, bool *is_success) {
  CeedDebug(ceed, "Running command:\n$ %s", command);
  FILE *output_stream = popen((command + std::string(" 2>&1")).c_str(), "r");

  CeedCheck(output_stream != nullptr, ceed, CEED_ERROR_BACKEND, "Failed to %s\ncommand:\n$ %s", message, command);

  char        line[CEED_MAX_RESOURCE_LEN] = "";
  std::string output                      = "";

  while (fgets(line, sizeof(line), output_stream) != nullptr) output += line;
  CeedDebug(ceed, "output:\n%s\n", output.c_str());
  CeedInt ierr = pclose(output_stream);

  if (is_success) *is_success = ierr == 0;
  if (err_on_fail) CeedCheck(ierr == 0, ceed, CEED_ERROR_BACKEND, "Failed to %s\ncommand:\n$ %s\nerror:\n%s", message, command, output.c_str());
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Build array of JIT flags
//------------------------------------------------------------------------------
static inline int CeedJitGetOpts_Cpu(Ceed ceed, const char ***opts, int *num_opts) {
  int opts_count = 2;

  // Standard options
  CeedCallBackend(CeedCalloc(opts_count, opts));
  // TODO: Revert
  // CeedCallBackend(CeedStringAllocCopy("-march=native", (char **)&(*opts)[0]));
  // CeedCallBackend(CeedStringAllocCopy("-O3", (char **)&(*opts)[1]));
  CeedCallBackend(CeedStringAllocCopy("-g", (char **)&(*opts)[0]));
  CeedCallBackend(CeedStringAllocCopy("-O0", (char **)&(*opts)[1]));

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

  // Compile kernel
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- ATTEMPTING TO COMPILE JIT SOURCE ----------\n");
  CeedDebug(ceed, "Name:\n  %s\n", name);
  CeedDebug(ceed, "Source:\n%s\n", code.str().c_str());
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- END OF JIT SOURCE ----------\n");

  {
    std::random_device         r;
    std::default_random_engine gen(r());
    // Place lower bound for uniformity of ids
    std::uniform_int_distribution<CeedInt> dist(1000000000);
    const CeedInt                          build_id      = dist(gen);
    std::string                            filename_base = std::string("temp/function_") + std::to_string(build_id) + "_" + name;

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
      std::string filename = filename_base + ".cpp";
      FILE       *file     = fopen(filename.c_str(), "w");

      CeedCheck(file, ceed, CEED_ERROR_BACKEND, "Failed to create file. Write access is required for cpu-jit");
      fputs(code.str().c_str(), file);
      fclose(file);
    }

    // Get CXX version
    Ceed_Cpu_Gen *ceed_data;

    CeedCallBackend(CeedGetData(ceed, &ceed_data));
    char *cxx = ceed_data->cxx;

    // First check for user JiT compiler
    if (!cxx) {
      const char *user_cxx = getenv("CEED_JIT_CXX");
      CeedDebug(ceed, "Attempting to detect user specified JiT compiler\nUser JiT compiler: %s\n", user_cxx);

      // Check if valid Clang
      bool is_valid = false;
      if (user_cxx) {
        std::string command = std::string(user_cxx) + " --version 2>&1";

        CeedDebug(ceed, "Checking user JiT compiler...");
        CeedCallSystem_Unchecked(ceed, command.c_str(), "checking user JiT compiler", &is_valid);
      }

      if (is_valid) {
        CeedDebug(ceed, "User specified JiT compiler is valid\n");
        CeedCall(CeedStringAllocCopy(user_cxx, &ceed_data->cxx));
        cxx = ceed_data->cxx;
      } else {
        CeedDebug(ceed, "Could not invoke user specified JiT compiler\n");
      }
    }
    // Default to c++
    if (!cxx) {
      CeedDebug(ceed, "Default JiT compiler: c++\n");
      CeedCall(CeedStringAllocCopy("c++", &ceed_data->cxx));
      cxx = ceed_data->cxx;
      {
        std::string command = std::string(cxx) + " --version 2>&1";

        CeedDebug(ceed, "Checking default JiT compiler...");
        CeedCallSystem_Unchecked(ceed, command.c_str(), "checking default JiT compiler", NULL);
      }
    }

    // Compile wrapper kernel
    std::string command = std::string(cxx) + " -shared -fPIC -rdynamic";

    for (CeedInt i = 0; i < num_opts; i++) command += std::string(" ") + opts[i];
    command += " " + filename_base + ".cpp -o " + filename_base + ".so";
    CeedCallSystem(ceed, command.c_str(), "JiT function source");
    CeedCallSystem(ceed, (std::string("chmod 0777 ") + filename_base + ".so").c_str(), "update JiT file permissions");

    // Load function from object file
    CeedDebug(ceed, (std::string("Loading object file: ") + filename_base + ".so").c_str());
    *handle          = dlopen((filename_base + ".so").c_str(), RTLD_NOW | RTLD_LOCAL);
    *is_compile_good = *handle != NULL;

    // Check load
    if (*is_compile_good) {
      void *function;

      CeedDebug(ceed, (std::string("Loading function: ") + name).c_str());
      function         = (void *)dlsym(*handle, name);
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
