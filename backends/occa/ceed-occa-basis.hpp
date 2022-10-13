// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_BASIS_HEADER
#define CEED_OCCA_BASIS_HEADER

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
namespace occa {
class Basis : public CeedObject {
 public:
  // Ceed object information
  CeedInt ceedComponentCount;

  // Owned information
  CeedInt dim;
  CeedInt P;
  CeedInt Q;

  Basis();

  virtual ~Basis();

  static Basis* getBasis(CeedBasis basis, const bool assertValid = true);

  static Basis* from(CeedBasis basis);
  static Basis* from(CeedOperatorField operatorField);

  int setCeedFields(CeedBasis basis);

  virtual bool isTensorBasis() const = 0;

  virtual const char* getFunctionSource() const = 0;

  virtual int apply(const CeedInt elementCount, CeedTransposeMode tmode, CeedEvalMode emode, Vector* u, Vector* v) = 0;

  //---[ Ceed Callbacks ]-----------
  static int registerCeedFunction(Ceed ceed, CeedBasis basis, const char* fname, ceed::occa::ceedFunction f);

  static int ceedApply(CeedBasis basis, const CeedInt nelem, CeedTransposeMode tmode, CeedEvalMode emode, CeedVector u, CeedVector v);

  static int ceedDestroy(CeedBasis basis);
};
}  // namespace occa
}  // namespace ceed

#endif
