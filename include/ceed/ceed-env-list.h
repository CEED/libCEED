/// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Header for declaring environment variables used by libCEED.
// No #pragma once because this file will be included twice to generate the necessary function declarations/definitions

#if defined(CEED_ENV_FLAG) && defined(CEED_ENV_STRING)
#ifndef CEED_ENV_ONLY_FLAG
#define UNSET_ONLY_FLAG
#define CEED_ENV_ONLY_FLAG(suffix, default, ...) CEED_ENV_FLAG(suffix, default, ##__VA_ARGS__)
#endif
#ifndef CEED_ENV_ONLY_STRING
#define UNSET_ONLY_STRING
#define CEED_ENV_ONLY_STRING(suffix, default, ...) CEED_ENV_STRING(suffix, default, ##__VA_ARGS__)
#endif

#define CEED_QUOTE(name) #name
#define CEED_STRINGIFY(macro) CEED_QUOTE(macro)

// Common environment variables
CEED_ENV_ONLY_FLAG(EnableDebug, false, "CEED_DEBUG", "DEBUG", "DBG")
CEED_ENV_ONLY_STRING(ErrorHandler, "abort", "CEED_ERROR_HANDLER")

// CPU backend
CEED_ENV_STRING(CpuJitCxx, CEED_STRINGIFY(CEED_CPU_JIT_CXX), "CEED_CPU_JIT_CXX")
CEED_ENV_STRING(CpuJitOpt, CEED_STRINGIFY(CEED_CPU_JIT_OPT), "CEED_CPU_JIT_OPT")

// HIP backend environment variables
CEED_ENV_ONLY_FLAG(HipHsaXnack, false, "HSA_XNACK")

// CUDA backend environment variables
CEED_ENV_FLAG(CudaEnableGraph, true, "CEED_CUDA_ENABLE_GRAPH", "CEED_ENABLE_CUDA_GRAPH")
CEED_ENV_FLAG(CudaUseClang, false, "CEED_CUDA_USE_CLANG", "CEED_USE_CLANG_CUDA")
CEED_ENV_STRING(CudaClangCxx, NULL, "CEED_CUDA_CLANG_CXX", "CEED_CLANG_CUDA_CXX")
CEED_ENV_STRING(CudaRustupToolchain, "nightly", "CEED_CUDA_RUSTUP_TOOLCHAIN", "RUSTUP_TOOLCHAIN")

#undef CEED_QUOTE
#undef CEED_STRINGIFY

#ifdef UNSET_ONLY_FLAG
#undef CEED_ENV_ONLY_FLAG
#undef UNSET_ONLY_FLAG
#endif
#ifdef UNSET_ONLY_STRING
#undef CEED_ENV_ONLY_STRING
#undef UNSET_ONLY_STRING
#endif
#endif
