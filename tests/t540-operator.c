/// @file
/// Test creation and use of FDM element inverse
/// \test Test creation and use of FDM element inverse
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t540-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictxi, Erestrictui, Erestrictqi;
  CeedBasis bx, bu;
  CeedQFunction qf_setup_mass, qf_apply;
  CeedOperator op_setup_mass, op_apply, op_inv;
  CeedVector qdata_mass, X, U, V;
  CeedInt nelem = 1, P = 4, Q = 5, dim = 2;
  CeedInt ndofs = P*P, nqpts = nelem*Q*Q;
  CeedScalar x[dim*nelem*(2*2)];
  const CeedScalar *u;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i=0; i<2; i++)
    for (CeedInt j=0; j<2; j++) {
      x[i+j*2+0*4] = i;
      x[i+j*2+1*4] = j;
    }
  CeedVectorCreate(ceed, dim*nelem*(2*2), &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector
  CeedVectorCreate(ceed, nqpts, &qdata_mass);

  // Element Setup

  // Restrictions
  CeedElemRestrictionCreateIdentity(ceed, nelem, 2*2, nelem*2*2, dim,
                                    &Erestrictxi);

  CeedElemRestrictionCreateIdentity(ceed, nelem, P*P, ndofs, 1, &Erestrictui);

  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q, nqpts, 1, &Erestrictqi);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &bu);

  // QFunction - setup mass
  CeedQFunctionCreateInterior(ceed, 1, setup_mass, setup_mass_loc,
                              &qf_setup_mass);
  CeedQFunctionAddInput(qf_setup_mass, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_mass, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_mass, "qdata", 1, CEED_EVAL_NONE);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", Erestrictxi, CEED_NOTRANSPOSE, bx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", Erestrictqi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup_mass, X, qdata_mass, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_apply, "qdata_mass", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_apply);
  CeedOperatorSetField(op_apply, "u", Erestrictui, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata_mass", Erestrictqi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata_mass);
  CeedOperatorSetField(op_apply, "v", Erestrictui, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);

  // Apply original operator
  CeedVectorCreate(ceed, ndofs, &U);
  CeedVectorSetValue(U, 1.0);
  CeedVectorCreate(ceed, ndofs, &V);
  CeedVectorSetValue(V, 0.0);
  CeedOperatorApply(op_apply, U, V, CEED_REQUEST_IMMEDIATE);

  // Create FDM element inverse
  CeedOperatorCreateFDMElementInverse(op_apply, &op_inv, CEED_REQUEST_IMMEDIATE);

  // Apply FDM element inverse
  CeedOperatorApply(op_inv, V, U, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u);
  for (int i=0; i<ndofs; i++)
    if (fabs(u[i] - 1.0) > 1E-14)
      // LCOV_EXCL_START
      printf("[%d] Error in inverse: %f != 1.0\n", i, u[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(U, &u);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_inv);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedElemRestrictionDestroy(&Erestrictqi);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&qdata_mass);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
