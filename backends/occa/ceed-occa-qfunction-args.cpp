// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-qfunction-args.hpp"

namespace ceed {
namespace occa {
QFunctionArgs::QFunctionArgs() : _isValid(false), _inputCount(0), _outputCount(0) {}

QFunctionArgs::QFunctionArgs(CeedQFunction qf) : _isValid(false), _inputCount(0), _outputCount(0) { setupQFunctionArgs(qf); }

void QFunctionArgs::setupQFunctionArgs(CeedQFunction qf) {
  CeedQFunctionField *ceedInputFields, *ceedOutputFields;

  CeedCallOccaValid(_isValid, CeedQFunctionGetCeed(qf, &ceed));

  CeedCallOccaValid(_isValid, CeedQFunctionGetNumArgs(qf, &_inputCount, &_outputCount));

  CeedCallOccaValid(_isValid, CeedQFunctionGetFields(qf, NULL, &ceedInputFields, NULL, &ceedOutputFields));

  _isValid = true;

  for (int i = 0; i < _inputCount; ++i) {
    QFunctionField field = QFunctionField(ceedInputFields[i]);
    qfInputs.push_back(field);
    _isValid &= field.isValid();
  }

  for (int i = 0; i < _outputCount; ++i) {
    QFunctionField field = QFunctionField(ceedOutputFields[i]);
    qfOutputs.push_back(field);
    _isValid &= field.isValid();
  }
}

bool QFunctionArgs::isValid() const { return _isValid; }

int QFunctionArgs::inputCount() const { return _inputCount; }

int QFunctionArgs::outputCount() const { return _outputCount; }

const QFunctionField& QFunctionArgs::getQfField(const bool isInput, const int index) const { return isInput ? qfInputs[index] : qfOutputs[index]; }

const QFunctionField& QFunctionArgs::getQfInput(const int index) const { return qfInputs[index]; }

const QFunctionField& QFunctionArgs::getQfOutput(const int index) const { return qfOutputs[index]; }

CeedEvalMode QFunctionArgs::getEvalMode(const bool isInput, const int index) const {
  return isInput ? qfInputs[index].evalMode : qfOutputs[index].evalMode;
}

CeedEvalMode QFunctionArgs::getInputEvalMode(const int index) const { return qfInputs[index].evalMode; }

CeedEvalMode QFunctionArgs::getOutputEvalMode(const int index) const { return qfOutputs[index].evalMode; }
}  // namespace occa
}  // namespace ceed
