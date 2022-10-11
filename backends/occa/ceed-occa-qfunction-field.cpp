// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-qfunction-field.hpp"

namespace ceed {
  namespace occa {
    QFunctionField::QFunctionField(CeedQFunctionField qfField) :
        _isValid(false),
        size(0) {

      int ierr = 0;

      ierr = CeedQFunctionFieldGetEvalMode(qfField, &evalMode);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedQFunctionFieldGetSize(qfField, &size);
      CeedOccaValidChk(_isValid, ierr);

      _isValid = true;
    }

    bool QFunctionField::isValid() const {
      return _isValid;
    }
  }
}
