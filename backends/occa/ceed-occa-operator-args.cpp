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

#include "ceed-occa-operator-args.hpp"

namespace ceed {
  namespace occa {
    OperatorArgs::OperatorArgs() :
        QFunctionArgs() {}

    OperatorArgs::OperatorArgs(CeedOperator op) :
        QFunctionArgs() {

      setupArgs(op);
    }

    void OperatorArgs::setupArgs(CeedOperator op) {
      CeedQFunction qf;
      CeedOperatorField *ceedInputFields, *ceedOutputFields;
      int ierr = 0;

      ierr = CeedOperatorGetQFunction(op, &qf);
      CeedOccaValidChk(_isValid, ierr);
      setupQFunctionArgs(qf);

      if (!_isValid) {
        return;
      }

      ierr = CeedOperatorGetFields(op, NULL, &ceedInputFields, NULL, &ceedOutputFields);
      CeedOccaValidChk(_isValid, ierr);

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

    const OperatorField& OperatorArgs::getOpField(const bool isInput,
                                                  const int index) const {
      return isInput ? opInputs[index] : opOutputs[index];
    }

    const OperatorField& OperatorArgs::getOpInput(const int index) const {
      return opInputs[index];
    }

    const OperatorField& OperatorArgs::getOpOutput(const int index) const {
      return opOutputs[index];
    }
  }
}
