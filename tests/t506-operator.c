/// @file
/// Test creation reuse of the same QFunction for multiple operators
/// \test Test creation reuse of the same QFunction for multiple operators
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictui_small, Erestrictui_large;
  CeedBasis bx_small, bx_large, bu_small, bu_large;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup_small, op_mass_small,
               op_setup_large, op_mass_large;
  CeedVector qdata_small, qdata_large, X, U, V;
  CeedScalar *hu;
  const CeedScalar *hv;
  CeedInt nelem = 15, P = 5, Q = 8, scale = 3;
  CeedInt Nx = nelem+1, Nu = nelem*(P-1)+1;
  CeedInt indx[nelem*2], indu[nelem*P];
  CeedScalar x[Nx];
  CeedScalar sum1, sum2;

  CeedInit(argv[1], &ceed);
  for (CeedInt i=0; i<Nx; i++) x[i] = (CeedScalar) i / (Nx - 1);
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
  CeedInt stridesu_small[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q, 1, Q*nelem, stridesu_small,
                                   &Erestrictui_small);
  CeedInt stridesu_large[3] = {1, Q*scale, Q*scale};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*scale, 1, Q*nelem*scale,
                                   stridesu_large, &Erestrictui_large);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q, CEED_GAUSS, &bu_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q*scale, CEED_GAUSS, &bx_large);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q*scale, CEED_GAUSS, &bu_large);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "x", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 2, CEED_EVAL_INTERP);

  // Input vector
  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // 'Small' Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup_small);
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_mass_small);

  CeedVectorCreate(ceed, nelem*Q, &qdata_small);

  CeedOperatorSetField(op_setup_small, "_weight", CEED_ELEMRESTRICTION_NONE,
                       bx_small, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_small, "x", Erestrictx,
                       bx_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_small, "rho", Erestrictui_small,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass_small, "rho", Erestrictui_small,
                       CEED_BASIS_COLLOCATED, qdata_small);
  CeedOperatorSetField(op_mass_small, "u", Erestrictu,
                       bu_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_small, "v", Erestrictu,
                       bu_small, CEED_VECTOR_ACTIVE);

  // 'Large' operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup_large);
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_mass_large);

  CeedVectorCreate(ceed, nelem*Q*scale, &qdata_large);

  CeedOperatorSetField(op_setup_large, "_weight", CEED_ELEMRESTRICTION_NONE,
                       bx_large, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_large, "x", Erestrictx,
                       bx_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_large, "rho", Erestrictui_large,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass_large, "rho", Erestrictui_large,
                       CEED_BASIS_COLLOCATED, qdata_large);
  CeedOperatorSetField(op_mass_large, "u", Erestrictu,
                       bu_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_large, "v", Erestrictu,
                       bu_large, CEED_VECTOR_ACTIVE);

  // Setup
  CeedOperatorApply(op_setup_small, X, qdata_small, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_large, X, qdata_large, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, 2*Nu, &U);
  CeedVectorGetArray(U, CEED_MEM_HOST, &hu);
  for (int i = 0; i < Nu; i++) {
    hu[2*i] = 1.0;
    hu[2*i+1] = 2.0;
  }
  CeedVectorRestoreArray(U, &hu);
  CeedVectorCreate(ceed, 2*Nu, &V);

  // 'Small' operator
  CeedOperatorApply(op_mass_small, U, V, CEED_REQUEST_IMMEDIATE);

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

  // 'Large' operator
  CeedOperatorApply(op_mass_large, U, V, CEED_REQUEST_IMMEDIATE);

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
  CeedOperatorDestroy(&op_setup_small);
  CeedOperatorDestroy(&op_mass_small);
  CeedOperatorDestroy(&op_setup_large);
  CeedOperatorDestroy(&op_mass_large);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui_small);
  CeedElemRestrictionDestroy(&Erestrictui_large);
  CeedBasisDestroy(&bu_small);
  CeedBasisDestroy(&bx_small);
  CeedBasisDestroy(&bu_large);
  CeedBasisDestroy(&bx_large);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&qdata_small);
  CeedVectorDestroy(&qdata_large);
  CeedDestroy(&ceed);
  return 0;
}
