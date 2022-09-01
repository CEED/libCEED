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

#ifndef CEED_OCCA_CEEDOBJECT_HEADER
#define CEED_OCCA_CEEDOBJECT_HEADER

#include "ceed-occa-context.hpp"

namespace ceed {
namespace occa {
class CeedObject {
 private:
  ::occa::device _device;

 public:
  Ceed ceed;

  CeedObject(Ceed ceed_ = NULL);

  ::occa::device getDevice();

  bool usingCpuDevice() const;
  bool usingGpuDevice() const;

  int        ceedError(const std::string &message) const;
  static int staticCeedError(const std::string &message);
};

namespace SyncState {
static const int none   = 0;
static const int host   = (1 << 0);
static const int device = (1 << 1);
static const int all    = host | device;
}  // namespace SyncState
}  // namespace occa
}  // namespace ceed

#endif
