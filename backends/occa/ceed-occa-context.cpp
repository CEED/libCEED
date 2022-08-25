// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

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
