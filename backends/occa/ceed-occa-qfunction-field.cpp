// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-qfunction-field.hpp"

namespace ceed {
namespace occa {
QFunctionField::QFunctionField(CeedQFunctionField qfField) : _isValid(false), size(0) {
  CeedCallOccaValid(_isValid, CeedQFunctionFieldGetEvalMode(qfField, &evalMode));

  CeedCallOccaValid(_isValid, CeedQFunctionFieldGetSize(qfField, &size));

  _isValid = true;
}

bool QFunctionField::isValid() const { return _isValid; }
}  // namespace occa
}  // namespace ceed
