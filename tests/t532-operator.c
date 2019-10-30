/// @file
/// Test assembly of mass and Poisson operator QFunction
/// \test Test assembly of mass and Poisson operator QFunction
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t532-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictxi, Erestrictui,
                      Erestrictqi, Erestrictlini;
  CeedBasis bx, bu;
  CeedQFunction qf_setup_mass, qf_setup_diff, qf_apply, qf_apply_lin;
  CeedOperator op_setup_mass, op_setup_diff, op_apply, op_apply_lin;
  CeedVector qdata_mass, qdata_diff, X, A, u, v;
  CeedInt nelem = 6, P = 3, Q = 4, dim = 2;
  CeedInt nx = 3, ny = 2;
  CeedInt ndofs = (nx*2+1)*(ny*2+1), nqpts = nelem*Q*Q;
  CeedInt indx[nelem*P*P];
  CeedScalar x[dim*ndofs];

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i=0; i<nx*2+1; i++)
    for (CeedInt j=0; j<ny*2+1; j++) {
      x[i+j*(nx*2+1)+0*ndofs] = (CeedScalar) i / (2*nx);
      x[i+j*(nx*2+1)+1*ndofs] = (CeedScalar) j / (2*ny);
    }
  CeedVectorCreate(ceed, dim*ndofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vectors
  CeedVectorCreate(ceed, nqpts, &qdata_mass);
  CeedVectorCreate(ceed, nqpts*dim*(dim+1)/2, &qdata_diff);

  // Element Setup
  for (CeedInt i=0; i<nelem; i++) {
    CeedInt col, row, offset;
    col = i % nx;
    row = i / nx;
    offset = col*(P-1) + row*(nx*2+1)*(P-1);
    for (CeedInt j=0; j<P; j++)
      for (CeedInt k=0; k<P; k++)
        indx[P*(P*i+k)+j] = offset + k*(nx*2+1) + j;
  }

  // Restrictions
  CeedElemRestrictionCreate(ceed, nelem, P*P, ndofs, dim, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);
  CeedElemRestrictionCreateIdentity(ceed, nelem, P*P, nelem*P*P, dim,
                                    &Erestrictxi);

  CeedElemRestrictionCreate(ceed, nelem, P*P, ndofs, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictu);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q, nqpts, 1, &Erestrictui);

  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q, nqpts, dim*(dim+1)/2,
                                    &Erestrictqi);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &bu);

  // QFunction - setup mass
  CeedQFunctionCreateInterior(ceed, 1, setup_mass, setup_mass_loc,
                              &qf_setup_mass);
  CeedQFunctionAddInput(qf_setup_mass, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_mass, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_mass, "qdata", 1, CEED_EVAL_NONE);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setup_mass, NULL, NULL, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", Erestrictx, CEED_NOTRANSPOSE, bx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diff
  CeedQFunctionCreateInterior(ceed, 1, setup_diff, setup_diff_loc,
                              &qf_setup_diff);
  CeedQFunctionAddInput(qf_setup_diff, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_diff, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_diff, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);

  // Operator - setup diff
  CeedOperatorCreate(ceed, qf_setup_diff, NULL, NULL, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", Erestrictx, CEED_NOTRANSPOSE, bx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "qdata", Erestrictqi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setup_mass, X, qdata_mass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_diff, X, qdata_diff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, apply, apply_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "qdata_mass", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "qdata_diff", dim*(dim+1)/2, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_apply, NULL, NULL, &op_apply);
  CeedOperatorSetField(op_apply, "du", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata_mass", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata_mass);
  CeedOperatorSetField(op_apply, "qdata_diff", Erestrictqi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata_diff);
  CeedOperatorSetField(op_apply, "u", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "dv", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);

  // Apply original operator
  CeedVectorCreate(ceed, ndofs, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, ndofs, &v);
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedScalar area = 0.0;
  const CeedScalar *vv;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i=0; i<ndofs; i++)
    area += vv[i];
  CeedVectorRestoreArrayRead(v, &vv);
  if (fabs(area - 1.0) > 1e-14)
    // LCOV_EXCL_START
    printf("Error: True operator computed area = %f != 1.0\n", area);
  // LCOV_EXCL_STOP

  // Assemble QFunction
  CeedOperatorAssembleLinearQFunction(op_apply, &A, &Erestrictlini,
                                      CEED_REQUEST_IMMEDIATE);

  // QFunction - apply assembled
  CeedQFunctionCreateInterior(ceed, 1, apply_lin, apply_lin_loc, &qf_apply_lin);
  CeedQFunctionAddInput(qf_apply_lin, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply_lin, "qdata", (dim+1)*(dim+1), CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_apply_lin, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply_lin, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_apply_lin, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply assembled
  CeedOperatorCreate(ceed, qf_apply_lin, NULL, NULL, &op_apply_lin);
  CeedOperatorSetField(op_apply_lin, "du", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_lin, "qdata", Erestrictlini, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, A);
  CeedOperatorSetField(op_apply_lin, "u", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_lin, "v", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply_lin, "dv", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);

  // Apply assembled QFunction operator
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_apply_lin, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  area = 0.0;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i=0; i<ndofs; i++)
    area += vv[i];
  CeedVectorRestoreArrayRead(v, &vv);
  if (fabs(area - 1.0) > 1e-14)
    // LCOV_EXCL_START
    printf("Error: Assembled operator computed area = %f != 1.0\n", area);
  // LCOV_EXCL_STOP

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedQFunctionDestroy(&qf_apply_lin);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedOperatorDestroy(&op_apply_lin);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedElemRestrictionDestroy(&Erestrictqi);
  CeedElemRestrictionDestroy(&Erestrictlini);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&qdata_mass);
  CeedVectorDestroy(&qdata_diff);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
