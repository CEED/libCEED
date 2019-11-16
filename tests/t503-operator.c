/// @file
/// Test creation, action, and destruction for mass matrix operator with passive inputs and outputs
/// \test Test creation, action, and destruction for mass matrix operator with passive inputs and outputs
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
  CeedScalar sum;

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, Nu, &U);
  CeedVectorSetValue(U, 1.0);

  CeedVectorCreate(ceed, Nu, &V);

  for (CeedInt i=0; i<Nx; i++)
    x[i] = (CeedScalar) i / (Nx - 1);
  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  CeedVectorCreate(ceed, nelem*Q, &qdata);

  // Restrictions
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed, nelem, 2, Nx, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);
  CeedElemRestrictionCreateIdentity(ceed, nelem, 2, nelem*2, 1, &Erestrictxi);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<P; j++) {
      indu[P*i+j] = i*(P-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, P, Nu, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indu, &Erestrictu);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q, Q*nelem, 1, &Erestrictui);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &bu);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_mass);

  CeedOperatorSetField(op_setup, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE, bx, X);
  CeedOperatorSetField(op_setup, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);

  CeedOperatorSetField(op_mass, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "u", Erestrictu, CEED_NOTRANSPOSE, bu, U);
  CeedOperatorSetField(op_mass, "v", Erestrictu, CEED_NOTRANSPOSE, bu, V);

  // Note - It is atypical to use only passive fields; this test is intended
  //   as a test for all passive input modes rather than as an example.
  CeedOperatorApply(op_setup, CEED_VECTOR_NONE, CEED_VECTOR_NONE,
                    CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_mass, CEED_VECTOR_NONE, CEED_VECTOR_NONE,
                    CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i=0; i<Nu; i++)
    sum += hv[i];
  if (fabs(sum-1.)>1e-10) printf("Computed Area: %f != True Area: 1.0\n", sum);
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
