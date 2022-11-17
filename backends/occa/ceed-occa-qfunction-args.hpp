// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_QFUNCTIONARGS_HEADER
#define CEED_OCCA_QFUNCTIONARGS_HEADER

#include <vector>

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-qfunction-field.hpp"

namespace ceed {
namespace occa {
typedef std::vector<QFunctionField> QFunctionFieldVector;

class QFunctionArgs : public CeedObject {
 protected:
  bool    _isValid;
  CeedInt _inputCount;
  CeedInt _outputCount;

 public:
  QFunctionFieldVector qfInputs;
  QFunctionFieldVector qfOutputs;

  QFunctionArgs();
  QFunctionArgs(CeedQFunction qf);

  void setupQFunctionArgs(CeedQFunction qf);

  bool isValid() const;

  int inputCount() const;
  int outputCount() const;

  const QFunctionField& getQfField(const bool isInput, const int index) const;

  const QFunctionField& getQfInput(const int index) const;

  const QFunctionField& getQfOutput(const int index) const;

  CeedEvalMode getEvalMode(const bool isInput, const int index) const;

  CeedEvalMode getInputEvalMode(const int index) const;

  CeedEvalMode getOutputEvalMode(const int index) const;
};
}  // namespace occa
}  // namespace ceed

#endif
