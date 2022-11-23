// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_QFUNCTIONFIELD_HEADER
#define CEED_OCCA_QFUNCTIONFIELD_HEADER

#include "ceed-occa-context.hpp"

namespace ceed {
namespace occa {
class QFunctionField {
 protected:
  bool _isValid;

 public:
  CeedEvalMode evalMode;
  CeedInt      size;

  QFunctionField(CeedQFunctionField qfField);

  bool isValid() const;
};
}  // namespace occa
}  // namespace ceed

#endif
