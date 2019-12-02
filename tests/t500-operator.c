/// @file
/// Test creation, action, and destruction for mass matrix operator
/// \test Test creation, action, and destruction for mass matrix operator
#include <ceed.h>
#include <stdlib.h>
#include <math.h>

#include "t500-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, U, V;
  const CeedScalar *hv;
  CeedInt nelem = 15, P = 5, Q = 8;
  CeedInt Nx = nelem+1, Nu = nelem*(P-1)+1;
  CeedInt indx[nelem*2], indu[nelem*P];
  CeedScalar x[Nx];

//! [Ceed Init]
  CeedInit(argv[1], &ceed);
//! [Ceed Init]
  for (CeedInt i=0; i<Nx; i++)
    x[i] = (CeedScalar) i / (Nx - 1);
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
//! [ElemRestr Create]
  CeedElemRestrictionCreate(ceed, nelem, 2, Nx, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);
  CeedElemRestrictionCreateIdentity(ceed, nelem, 2, nelem*2, 1, &Erestrictxi);
//! [ElemRestr Create]

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<P; j++) {
      indu[P*i+j] = i*(P-1) + j;
    }
  }
//! [ElemRestrU Create]
  CeedElemRestrictionCreate(ceed, nelem, P, Nu, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indu, &Erestrictu);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q, Q*nelem, 1, &Erestrictui);
//! [ElemRestrU Create]

//! [Basis Create]
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &bu);
//! [Basis Create]

//! [QFunction Create]
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);
//! [QFunction Create]

//! [Setup Create]
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup);
//! [Setup Create]

//! [Operator Create]
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_mass);
//! [Operator Create]

  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorCreate(ceed, nelem*Q, &qdata);

//! [Setup Set]
  CeedOperatorSetField(op_setup, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
//! [Setup Set]

//! [Operator Set]
  CeedOperatorSetField(op_mass, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "u", Erestrictu, CEED_NOTRANSPOSE,
                       bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", Erestrictu, CEED_NOTRANSPOSE,
                       bu, CEED_VECTOR_ACTIVE);
//! [Operator Set]

//! [Setup Apply]
  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);
//! [Setup Apply]

  CeedVectorCreate(ceed, Nu, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, Nu, &V);
//! [Operator Apply]
  CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);
//! [Operator Apply]

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  for (CeedInt i=0; i<Nu; i++)
    if (fabs(hv[i]) > 1e-14) printf("[%d] v %g != 0.0\n",i, hv[i]);
  CeedVectorRestoreArrayRead(V, &hv);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&qdata);
  CeedDestroy(&ceed);
  return 0;
}
