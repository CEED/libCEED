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

  void                  setupQFunctionArgs(CeedQFunction qf);

  bool                  isValid() const;

  int                   inputCount() const;
  int                   outputCount() const;

  const QFunctionField& getQfField(const bool isInput, const int index) const;

  const QFunctionField& getQfInput(const int index) const;

  const QFunctionField& getQfOutput(const int index) const;

  CeedEvalMode          getEvalMode(const bool isInput, const int index) const;

  CeedEvalMode          getInputEvalMode(const int index) const;

  CeedEvalMode          getOutputEvalMode(const int index) const;
};
}  // namespace occa
}  // namespace ceed

#endif
