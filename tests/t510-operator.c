/// @file
/// Test creation creation, action, and destruction for mass matrix operator
/// \test Test creation creation, action, and destruction for mass matrix operator
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t310-basis.h"

#include "t510-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, U, V;
  const CeedScalar *hv;
  CeedInt nelem = 12, dim = 2, P = 6, Q = 4;
  CeedInt nx = 3, ny = 2;
  CeedInt row, col, offset;
  CeedInt Ndofs = (nx*2+1)*(ny*2+1), Nqpts = nelem*Q;
  CeedInt indx[nelem*P];
  CeedScalar x[dim*Ndofs];
  CeedScalar qref[dim*Q], qweight[Q];
  CeedScalar interp[P*Q], grad[dim*P*Q];

  CeedInit(argv[1], &ceed);

  for (CeedInt i=0; i<Ndofs; i++) {
    x[i] = (1. / (nx*2)) * (CeedScalar) (i % (nx*2+1));
    x[i+Ndofs] = (1. / (ny*2)) * (CeedScalar) (i / (nx*2+1));
  }
  for (CeedInt i=0; i<nelem/2; i++) {
    col = i % nx;
    row = i / nx;
    offset = col*2 + row*(nx*2+1)*2;

    indx[i*2*P +  0] =  2 + offset;
    indx[i*2*P +  1] =  9 + offset;
    indx[i*2*P +  2] = 16 + offset;
    indx[i*2*P +  3] =  1 + offset;
    indx[i*2*P +  4] =  8 + offset;
    indx[i*2*P +  5] =  0 + offset;

    indx[i*2*P +  6] = 14 + offset;
    indx[i*2*P +  7] =  7 + offset;
    indx[i*2*P +  8] =  0 + offset;
    indx[i*2*P +  9] = 15 + offset;
    indx[i*2*P + 10] =  8 + offset;
    indx[i*2*P + 11] = 16 + offset;
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, nelem, P, Ndofs, dim, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);
  CeedElemRestrictionCreateIdentity(ceed, nelem, P, nelem*P, dim, &Erestrictxi);

  CeedElemRestrictionCreate(ceed, nelem, P, Ndofs, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictu);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q, Nqpts, 1, &Erestrictui);


  // Bases
  buildmats(qref, qweight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TRIANGLE, dim, P, Q, interp, grad, qref,
                    qweight, &bx);

  buildmats(qref, qweight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TRIANGLE, 1, P, Q, interp, grad, qref,
                    qweight, &bu);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);

  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);

  CeedVectorCreate(ceed, dim*Ndofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorCreate(ceed, Nqpts, &qdata);

  CeedOperatorSetField(op_setup, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "u", Erestrictu, CEED_NOTRANSPOSE,
                       bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", Erestrictu, CEED_NOTRANSPOSE,
                       bu, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, Ndofs, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, Ndofs, &V);

  CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  for (CeedInt i=0; i<Ndofs; i++)
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
