// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_TENSORBASIS_HEADER
#define CEED_OCCA_TENSORBASIS_HEADER

#include "ceed-occa-basis.hpp"

namespace ceed {
namespace occa {
class TensorBasis : public Basis {
 public:
  CeedInt        P1D;
  CeedInt        Q1D;
  ::occa::memory interp1D;
  ::occa::memory grad1D;
  ::occa::memory qWeight1D;

  ::occa::json   kernelProperties;
  ::occa::kernel interpKernel;
  ::occa::kernel interpTKernel;
  ::occa::kernel gradKernel;
  ::occa::kernel gradTKernel;
  ::occa::kernel weightKernel;

  TensorBasis(CeedBasis basis, CeedInt dim_, CeedInt P1D_, CeedInt Q1D_, const CeedScalar *interp1D_, const CeedScalar *grad1D_,
              const CeedScalar *qWeight1D_);

  ~TensorBasis();

  bool isTensorBasis() const;

  const char *getFunctionSource() const;

  std::string getKernelSource() const;

  void setKernelProperties();

  int elementsPerBlockInterp() const;
  int elementsPerBlockGrad() const;
  int elementsPerBlockWeight() const;

  ::occa::kernel buildKernel(const std::string &kernelName);

  int applyInterp(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V);

  int applyGrad(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V);

  int applyWeight(const CeedInt elementCount, Vector &W);

  int apply(const CeedInt elementCount, CeedTransposeMode tmode, CeedEvalMode emode, Vector *U, Vector *V);

  //---[ Ceed Callbacks ]-----------
  static int ceedCreate(CeedInt dim, CeedInt P1D, CeedInt Q1D, const CeedScalar *interp1D, const CeedScalar *grad1D, const CeedScalar *qref1D,
                        const CeedScalar *qWeight1D, CeedBasis basis);
};
}  // namespace occa
}  // namespace ceed

#endif
