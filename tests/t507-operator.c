/// @file
/// Test VLA macro for operator
/// \test VLA marco for operator
#include <ceed.h>
#include <stdlib.h>
#include <math.h>

#include "t507-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictui;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, U, V;
  CeedScalar *hu;
  const CeedScalar *hv;
  CeedInt nelem = 15, P = 5, Q = 8;
  CeedInt Nx = nelem+1, Nu = nelem*(P-1)+1;
  CeedInt indx[nelem*2], indu[nelem*P];
  CeedScalar x[Nx];
  CeedScalar sum1, sum2;

  CeedInit(argv[1], &ceed);

  for (CeedInt i=0; i<Nx; i++)
    x[i] = (CeedScalar) i / (Nx - 1);
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
  // Restrictions
  CeedElemRestrictionCreate(ceed, nelem, 2, 1, 1, Nx, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<P; j++) {
      indu[P*i+j] = 2*(i*(P-1) + j);
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, P, 2, 1, 2*Nu, CEED_MEM_HOST,
                            CEED_USE_POINTER, indu, &Erestrictu);
  CeedInt stridesu[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q, 1, Q*nelem, stridesu,
                                   &Erestrictui);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q, CEED_GAUSS, &bu);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1*1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 2, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_mass);

  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorCreate(ceed, nelem*Q, &qdata);

  CeedOperatorSetField(op_setup, "_weight", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", Erestrictui, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass, "rho", Erestrictui, CEED_BASIS_COLLOCATED,
                       qdata);
  CeedOperatorSetField(op_mass, "u", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", Erestrictu, bu, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, 2*Nu, &U);
  CeedVectorGetArray(U, CEED_MEM_HOST, &hu);
  for (int i = 0; i < Nu; i++) {
    hu[2*i] = 1.0;
    hu[2*i+1] = 2.0;
  }
  CeedVectorRestoreArray(U, &hu);
  CeedVectorCreate(ceed, 2*Nu, &V);
  CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum1 = 0.; sum2 = 0.;
  for (CeedInt i=0; i<Nu; i++) {
    sum1 += hv[2*i];
    sum2 += hv[2*i+1];
  }
  if (fabs(sum1-1.)>1e-10) printf("Computed Area: %f != True Area: 1.0\n", sum1);
  if (fabs(sum2-2.)>1e-10) printf("Computed Area: %f != True Area: 2.0\n", sum2);
  CeedVectorRestoreArrayRead(V, &hv);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&qdata);
  CeedDestroy(&ceed);
  return 0;
}
