// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_QFUNCTION_HEADER
#define CEED_OCCA_QFUNCTION_HEADER

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-qfunction-args.hpp"

namespace ceed {
namespace occa {
class QFunction : public CeedObject {
 public:
  // Ceed object information
  bool ceedIsIdentity;

  // Owned resources
  std::string          filename;
  std::string          qFunctionName;
  ::occa::kernel       qFunctionKernel;
  CeedQFunctionContext qFunctionContext;
  QFunctionArgs        args;

  QFunction(const std::string &source, const std::string &function_name);

  static QFunction *getQFunction(CeedQFunction qf, const bool assertValid = true);

  static QFunction *from(CeedQFunction qf);
  static QFunction *from(CeedOperator op);

  ::occa::properties getKernelProps(const CeedInt Q);

  int         buildKernel(const CeedInt Q);
  std::string getKernelSource(const std::string &kernelName, const CeedInt Q);

  int apply(CeedInt Q, CeedVector *U, CeedVector *V);

  //---[ Ceed Callbacks ]-----------
  static int registerCeedFunction(Ceed ceed, CeedQFunction qf, const char *fname, ceed::occa::ceedFunction f);

  static int ceedCreate(CeedQFunction qf);

  static int ceedApply(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V);

  static int ceedDestroy(CeedQFunction qf);
};
}  // namespace occa
}  // namespace ceed

#endif
