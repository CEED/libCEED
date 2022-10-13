// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_OPERATORARGS_HEADER
#define CEED_OCCA_OPERATORARGS_HEADER

#include <vector>

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-operator-field.hpp"
#include "ceed-occa-qfunction-args.hpp"

namespace ceed {
namespace occa {
typedef std::vector<OperatorField> OperatorFieldVector;

class OperatorArgs : public QFunctionArgs {
 public:
  OperatorFieldVector opInputs;
  OperatorFieldVector opOutputs;

  OperatorArgs();
  OperatorArgs(CeedOperator op);

  void setupArgs(CeedOperator op);

  const OperatorField& getOpField(const bool isInput, const int index) const;

  const OperatorField& getOpInput(const int index) const;

  const OperatorField& getOpOutput(const int index) const;
};
}  // namespace occa
}  // namespace ceed

#endif
