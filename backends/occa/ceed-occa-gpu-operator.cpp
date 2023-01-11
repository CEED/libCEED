// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-gpu-operator.hpp"

#include "ceed-occa-qfunction.hpp"

namespace ceed {
namespace occa {
GpuOperator::GpuOperator() {}

GpuOperator::~GpuOperator() {}

::occa::kernel GpuOperator::buildApplyAddKernel() { return ::occa::kernel(); }

void GpuOperator::applyAdd(Vector *in, Vector *out) {
  // TODO: Implement
}
}  // namespace occa
}  // namespace ceed
