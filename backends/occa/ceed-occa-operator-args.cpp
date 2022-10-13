// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-operator-args.hpp"

namespace ceed {
namespace occa {
OperatorArgs::OperatorArgs() : QFunctionArgs() {}

OperatorArgs::OperatorArgs(CeedOperator op) : QFunctionArgs() { setupArgs(op); }

void OperatorArgs::setupArgs(CeedOperator op) {
  CeedQFunction      qf;
  CeedOperatorField *ceedInputFields, *ceedOutputFields;

  CeedCallOccaValid(_isValid, CeedOperatorGetQFunction(op, &qf));
  setupQFunctionArgs(qf);

  if (!_isValid) {
    return;
  }

  CeedCallOccaValid(_isValid, CeedOperatorGetFields(op, NULL, &ceedInputFields, NULL, &ceedOutputFields));

  for (int i = 0; i < _inputCount; ++i) {
    OperatorField field = OperatorField(ceedInputFields[i]);
    opInputs.push_back(field);
    _isValid &= field.isValid();
  }

  for (int i = 0; i < _outputCount; ++i) {
    OperatorField field = OperatorField(ceedOutputFields[i]);
    opOutputs.push_back(field);
    _isValid &= field.isValid();
  }
}

const OperatorField& OperatorArgs::getOpField(const bool isInput, const int index) const { return isInput ? opInputs[index] : opOutputs[index]; }

const OperatorField& OperatorArgs::getOpInput(const int index) const { return opInputs[index]; }

const OperatorField& OperatorArgs::getOpOutput(const int index) const { return opOutputs[index]; }
}  // namespace occa
}  // namespace ceed
