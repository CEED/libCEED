// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_GPU_OPERATOR_HEADER
#define CEED_OCCA_GPU_OPERATOR_HEADER

#include <vector>

#include "ceed-occa-operator.hpp"

namespace ceed {
namespace occa {
class GpuOperator : public Operator {
 public:
  GpuOperator();

  ~GpuOperator();

  ::occa::kernel buildApplyAddKernel();

  void applyAdd(Vector *in, Vector *out);
};
}  // namespace occa
}  // namespace ceed

#endif
