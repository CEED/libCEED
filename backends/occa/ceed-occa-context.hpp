// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_CONTEXT_HEADER
#define CEED_OCCA_CONTEXT_HEADER

#include "ceed-occa-types.hpp"

namespace ceed {
namespace occa {
class Context {
 private:
  bool _usingCpuDevice;
  bool _usingGpuDevice;

 public:
  ::occa::device device;

  Context(::occa::device device_);

  static Context* from(Ceed ceed);

  bool usingCpuDevice() const;
  bool usingGpuDevice() const;
};
}  // namespace occa
}  // namespace ceed

#endif
