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

#ifndef CEED_OCCA_SIMPLEXBASIS_HEADER
#define CEED_OCCA_SIMPLEXBASIS_HEADER

#include "ceed-occa-basis.hpp"

namespace ceed {
namespace occa {
class SimplexBasis : public Basis {
 public:
  ::occa::memory        interp;
  ::occa::memory        grad;
  ::occa::memory        qWeight;
  ::occa::kernelBuilder interpKernelBuilder;
  ::occa::kernelBuilder gradKernelBuilder;
  ::occa::kernelBuilder weightKernelBuilder;

  SimplexBasis(CeedBasis basis, CeedInt dim, CeedInt P_, CeedInt Q_, const CeedScalar *interp_, const CeedScalar *grad_, const CeedScalar *qWeight_);

  ~SimplexBasis();

  bool isTensorBasis() const;

  const char *getFunctionSource() const;

  void setupKernelBuilders();

  int applyInterp(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V);

  ::occa::kernel getCpuInterpKernel(const bool transpose);
  ::occa::kernel getGpuInterpKernel(const bool transpose);

  int applyGrad(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V);

  ::occa::kernel getCpuGradKernel(const bool transpose);
  ::occa::kernel getGpuGradKernel(const bool transpose);

  int applyWeight(const CeedInt elementCount, Vector &W);

  ::occa::kernel getCpuWeightKernel();
  ::occa::kernel getGpuWeightKernel();

  ::occa::kernel buildCpuEvalKernel(::occa::kernelBuilder &kernelBuilder, const bool transpose);

  ::occa::kernel buildGpuEvalKernel(::occa::kernelBuilder &kernelBuilder, const bool transpose);

  int apply(const CeedInt elementCount, CeedTransposeMode tmode, CeedEvalMode emode, Vector *u, Vector *v);

  //---[ Ceed Callbacks ]-----------
  static int ceedCreate(CeedElemTopology topology, CeedInt dim, CeedInt ndof, CeedInt nquad, const CeedScalar *interp, const CeedScalar *grad,
                        const CeedScalar *qref, const CeedScalar *qWeight, CeedBasis basis);
};
}  // namespace occa
}  // namespace ceed

#endif
