// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-context.hpp"

namespace ceed {
namespace occa {
Context::Context(::occa::device device_) : device(device_) {
  const std::string mode = device.mode();
  _usingCpuDevice        = (mode == "Serial" || mode == "OpenMP");
  _usingGpuDevice        = (mode == "CUDA" || mode == "HIP" || mode == "OpenCL");
}

Context* Context::from(Ceed ceed) {
  if (!ceed) {
    return NULL;
  }

  Context* context;
  CeedGetData(ceed, (void**)&context);
  return context;
}

bool Context::usingCpuDevice() const { return _usingCpuDevice; }

bool Context::usingGpuDevice() const { return _usingGpuDevice; }
}  // namespace occa
}  // namespace ceed
