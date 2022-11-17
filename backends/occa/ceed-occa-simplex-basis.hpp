// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_SIMPLEXBASIS_HEADER
#define CEED_OCCA_SIMPLEXBASIS_HEADER

#include "ceed-occa-basis.hpp"

namespace ceed {
namespace occa {
class SimplexBasis : public Basis {
 public:
  ::occa::memory interp;
  ::occa::memory grad;
  ::occa::memory qWeight;

  ::occa::json   kernelProperties;
  ::occa::kernel interpKernel;
  ::occa::kernel interpTKernel;
  ::occa::kernel gradKernel;
  ::occa::kernel gradTKernel;
  ::occa::kernel weightKernel;

  SimplexBasis(CeedBasis basis, CeedInt dim, CeedInt P_, CeedInt Q_, const CeedScalar *interp_, const CeedScalar *grad_, const CeedScalar *qWeight_);

  ~SimplexBasis();

  bool isTensorBasis() const;

  const char *getFunctionSource() const;

  void setKernelProperties();

  std::string getKernelSource() const;

  ::occa::kernel buildKernel(const std::string &kernelName);

  int applyInterp(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V);

  int applyGrad(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V);

  int applyWeight(const CeedInt elementCount, Vector &W);

  int apply(const CeedInt elementCount, CeedTransposeMode tmode, CeedEvalMode emode, Vector *u, Vector *v);

  //---[ Ceed Callbacks ]-----------
  static int ceedCreate(CeedElemTopology topology, CeedInt dim, CeedInt ndof, CeedInt nquad, const CeedScalar *interp, const CeedScalar *grad,
                        const CeedScalar *qref, const CeedScalar *qWeight, CeedBasis basis);
};
}  // namespace occa
}  // namespace ceed

#endif
