// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-operator-field.hpp"

#include "ceed-occa-basis.hpp"
#include "ceed-occa-elem-restriction.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
namespace occa {
OperatorField::OperatorField(CeedOperatorField opField) : _isValid(false), _usesActiveVector(false), vec(NULL), basis(NULL), elemRestriction(NULL) {
  CeedBasis           ceedBasis;
  CeedVector          ceedVector;
  CeedElemRestriction ceedElemRestriction;

  CeedCallOccaValid(_isValid, CeedOperatorFieldGetBasis(opField, &ceedBasis));

  CeedCallOccaValid(_isValid, CeedOperatorFieldGetVector(opField, &ceedVector));

  CeedCallOccaValid(_isValid, CeedOperatorFieldGetElemRestriction(opField, &ceedElemRestriction));

  _isValid          = true;
  _usesActiveVector = ceedVector == CEED_VECTOR_ACTIVE;

  vec             = Vector::from(ceedVector);
  basis           = Basis::from(ceedBasis);
  elemRestriction = ElemRestriction::from(ceedElemRestriction);
}

bool OperatorField::isValid() const { return _isValid; }

//---[ Vector Info ]----------------
bool OperatorField::usesActiveVector() const { return _usesActiveVector; }
//==================================

//---[ Basis Info ]-----------------
bool OperatorField::hasBasis() const { return basis; }

int OperatorField::usingTensorBasis() const { return basis->isTensorBasis(); }

int OperatorField::getComponentCount() const { return (basis ? basis->ceedComponentCount : 1); }

int OperatorField::getP() const { return (basis ? basis->P : 0); }

int OperatorField::getQ() const { return (basis ? basis->Q : 0); }

int OperatorField::getDim() const { return (basis ? basis->dim : 1); }
//==================================

//---[ ElemRestriction Info ]-------
int OperatorField::getElementCount() const { return (elemRestriction ? elemRestriction->ceedElementCount : 1); }

int OperatorField::getElementSize() const { return (elemRestriction ? elemRestriction->ceedElementSize : 1); }
//==================================
}  // namespace occa
}  // namespace ceed
