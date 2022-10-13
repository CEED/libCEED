// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_OPERATOR_HEADER
#define CEED_OCCA_OPERATOR_HEADER

#include <vector>

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-operator-args.hpp"

namespace ceed {
namespace occa {
typedef std::vector<ceed::occa::Vector *> VectorVector_t;

class QFunction;

class Operator : public CeedObject {
 public:
  // Ceed object information
  CeedInt ceedQ;
  CeedInt ceedElementCount;

  // Owned resources
  QFunction     *qfunction;
  OperatorArgs   args;
  ::occa::kernel applyAddKernel;
  bool           needsInitialSetup;

  // Reference to other memory
  ::occa::memory qFunctionContextData;

  Operator();
  virtual ~Operator();

  static Operator *getOperator(CeedOperator op, const bool assertValid = true);

  static Operator *from(CeedOperator op);

  bool isApplyingIdentityFunction();

  int applyAdd(Vector *in, Vector *out, CeedRequest *request);

  //---[ Virtual Methods ]----------
  virtual ::occa::kernel buildApplyAddKernel() = 0;

  virtual void initialSetup();

  virtual void applyAdd(Vector *in, Vector *out) = 0;

  //---[ Ceed Callbacks ]-----------
  static int registerCeedFunction(Ceed ceed, CeedOperator op, const char *fname, ceed::occa::ceedFunction f);

  static int ceedCreate(CeedOperator op);
  static int ceedCreateComposite(CeedOperator op);

  static int ceedLinearAssembleQFunction(CeedOperator op);
  static int ceedLinearAssembleQFunctionUpdate(CeedOperator op);
  static int ceedLinearAssembleAddDiagonal(CeedOperator op);
  static int ceedLinearAssembleAddPointBlockDiagonal(CeedOperator op);
  static int ceedCreateFDMElementInverse(CeedOperator op);

  static int ceedApplyAdd(CeedOperator op, CeedVector invec, CeedVector outvec, CeedRequest *request);

  static int ceedDestroy(CeedOperator op);
};
}  // namespace occa
}  // namespace ceed

#endif
