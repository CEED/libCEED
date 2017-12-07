// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "ceed.h"
#include <stdlib.h>
#include <math.h>

static int f_mass(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u, CeedScalar *const *v) {
  const CeedScalar *w = qdata;
  for (CeedInt i=0; i<Q; i++) v[0][i] = w[i] * u[0][i];
  return 0;
}

static int f_poisson3d(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u, CeedScalar *const *v) {
  // Q is guaranteed to be a multiple of 8 (because of how we call CeedQFunctionCreateInterior) so we can tell the compiler
  Q = 8*(Q/8);
  // qdata can be packed arbitrarily, but we'll choose a vector-friendly ordering here
  const CeedScalar *rhs = qdata;
  const CeedScalar (*K)[Q] = (const CeedScalar(*)[Q])(rhs + Q);  // Probably symmetric but we don't have to exploit it
  for (CeedInt i=0; i<Q; i++) {
    v[0][i] = -rhs[i];
    for (CeedInt d=0; d<3; d++) {
      v[1][d*Q+i] = K[d*3+0][i] * u[1][0*Q+i] + K[d*3+1][i] * u[1][1*Q+i] + K[d*3+2][i] * u[1][2*Q+i];
    }
  }
  return 0;
}

static int f_buildcoeffs(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u, CeedScalar *const *v) {
  CeedScalar *rhs = qdata;
  CeedScalar (*K)[Q] = (CeedScalar(*)[Q])(rhs + Q);
  for (CeedInt i=0; i<Q; i++) {
    // RHS as an analytic function of the coordinates
    rhs[i] = cos(u[0][0*Q+i]) * cos(u[0][1*Q+i]) * cos(u[0][1*Q+i]);
    // ... set K using the gradient of coordinates u[1][...]
    for (CeedInt d=0; d<3; d++) {
      for (CeedInt e=0; e<3; e++) {
        K[d*3+e][i] = (u[1][(d*3+0)*Q+i] * u[1][(e*3+0)*Q+i] +
                       u[1][(d*3+1)*Q+i] * u[1][(e*3+1)*Q+i] +
                       u[1][(d*3+2)*Q+i] * u[1][(e*3+2)*Q+i]); // quadrature weight elided
      }
    }
  }
  return 0;
}

int main(int argc, char **argv)
{
  Ceed ceed;
  CeedVector u, r, xcoord, qdata;
  CeedInt nelem = 8, esize = 64, ndof = 343, *Eindices;
  CeedElemRestriction Erestrict;
  CeedBasis Basis;
  CeedQFunction qf_mass, qf_poisson3d, qf_buildcoeffs;
  CeedOperator op_mass, op_poisson3d, op_buildcoeffs;

  CeedInit("/cpu/self", &ceed); // implementation aborts on error by default
  CeedVectorCreate(ceed, ndof, &u);
  CeedVectorCreate(ceed, ndof, &r);
  CeedVectorCreate(ceed, ndof*3, &xcoord);

  Eindices = malloc(123 * 125 * sizeof(Eindices[0]));
  // call function to initialize Eindices...
  CeedElemRestrictionCreate(ceed, nelem, esize, ndof, CEED_MEM_HOST, CEED_USE_POINTER, Eindices, &Erestrict);

  // Create a 3D Q_3 Lagrange element with 4^3 Gauss quadrature points
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, 3, 4, CEED_GAUSS, &Basis);

  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar), CEED_EVAL_INTERP, CEED_EVAL_INTERP, f_mass, "ex1.c:f_mass", &qf_mass);
  CeedQFunctionCreateInterior(ceed, 8, 1, 10*sizeof(CeedScalar), CEED_EVAL_GRAD, CEED_EVAL_GRAD, f_poisson3d, "ex1.c:f_poisson3d", &qf_poisson3d);
  CeedQFunctionCreateInterior(ceed, 1, 3, 10*sizeof(CeedScalar), CEED_EVAL_INTERP | CEED_EVAL_GRAD, CEED_EVAL_NONE, f_buildcoeffs, "ex1.c:f_buildcoeffs", &qf_buildcoeffs);
  // We'll expect to build libraries of qfunctions, looked up by some name.  These should be cheap to create even if not used.

  CeedOperatorCreate(ceed, Erestrict, Basis, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorCreate(ceed, Erestrict, Basis, qf_poisson3d, NULL, NULL, &op_poisson3d);
  CeedOperatorCreate(ceed, Erestrict, Basis, qf_buildcoeffs, NULL, NULL, &op_buildcoeffs);

  // ... initialize xcoord

  // Apply the operator
  CeedOperatorGetQData(op_poisson3d, &qdata); // allocates if needed
  CeedOperatorApply(op_buildcoeffs, qdata, xcoord, NULL, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_poisson3d, qdata, u, r, CEED_REQUEST_IMMEDIATE);

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&r);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_poisson3d);
  CeedQFunctionDestroy(&qf_mass);
  CeedQFunctionDestroy(&qf_poisson3d);
  CeedBasisDestroy(&Basis);
  CeedElemRestrictionDestroy(&Erestrict);
  CeedDestroy(&ceed);
  return 0;
}
