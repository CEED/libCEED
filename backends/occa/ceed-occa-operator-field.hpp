// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_OPERATORFIELD_HEADER
#define CEED_OCCA_OPERATORFIELD_HEADER

#include "ceed-occa-context.hpp"

namespace ceed {
namespace occa {
class Basis;
class ElemRestriction;
class Vector;

class OperatorField {
 private:
  bool _isValid;
  bool _usesActiveVector;

 public:
  Vector          *vec;
  Basis           *basis;
  ElemRestriction *elemRestriction;

  OperatorField(CeedOperatorField opField);

  bool isValid() const;

  //---[ Vector Info ]--------------
  bool usesActiveVector() const;
  //================================

  //---[ Basis Info ]---------------
  bool hasBasis() const;
  int  usingTensorBasis() const;

  int getComponentCount() const;
  int getP() const;
  int getQ() const;
  int getDim() const;
  //================================

  //---[ ElemRestriction Info ]-----
  int getElementCount() const;
  int getElementSize() const;
  //================================
};
}  // namespace occa
}  // namespace ceed

#endif
