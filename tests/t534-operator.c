/// @file
/// Test assembly of Poisson operator QFunction
/// \test Test assembly of Poisson operator QFunction
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t534-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictxi, Erestrictui,
                      Erestrictqi;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_diff;
  CeedOperator op_setup, op_diff;
  CeedVector qdata, X, A, U, V;
  CeedInt nelem = 6, P = 3, Q = 4, dim = 2;
  CeedInt nx = 3, ny = 2;
  CeedInt ndofs = (nx*2+1)*(ny*2+1), nqpts = nelem*Q*Q;
  CeedInt indx[nelem*P*P];
  CeedScalar x[dim*ndofs], assembledTrue[ndofs];
  CeedScalar *u;
  const CeedScalar *a, *v;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i=0; i<nx*2+1; i++)
    for (CeedInt j=0; j<ny*2+1; j++) {
      x[i+j*(nx*2+1)+0*ndofs] = (CeedScalar) i / (2*nx);
      x[i+j*(nx*2+1)+1*ndofs] = (CeedScalar) j / (2*ny);
    }
  CeedVectorCreate(ceed, dim*ndofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vector
  CeedVectorCreate(ceed, nqpts*dim*(dim+1)/2, &qdata);

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

  // QFunction - setup
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);

  // Operator - setup
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE, bx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "_weight", Erestrictxi, CEED_NOTRANSPOSE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", Erestrictqi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, diff, diff_loc, &qf_diff);
  CeedQFunctionAddInput(qf_diff, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_diff, NULL, NULL, &op_diff);
  CeedOperatorSetField(op_diff, "du", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "qdata", Erestrictqi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_diff, "dv", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);

  // Assemble diagonal
  CeedOperatorAssembleLinearDiagonal(op_diff, &A, CEED_REQUEST_IMMEDIATE);

  // Manually assemble diagonal
  CeedVectorCreate(ceed, ndofs, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, ndofs, &V);
  for (int i=0; i<ndofs; i++) {
    // Set input
    CeedVectorGetArray(U, CEED_MEM_HOST, &u);
    u[i] = 1.0;
    if (i)
      u[i-1] = 0.0;
    CeedVectorRestoreArray(U, &u);

    // Compute diag entry for DoF i
    CeedOperatorApply(op_diff, U, V, CEED_REQUEST_IMMEDIATE);

    // Retrieve entry
    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    assembledTrue[i] = v[i];
    CeedVectorRestoreArrayRead(V, &v);
  }

  // Check output
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  for (int i=0; i<ndofs; i++)
    if (fabs(a[i] - assembledTrue[i]) > 1e-14)
      // LCOV_EXCL_START
      printf("[%d] Error in assembly: %f != %f\n", i, a[i], assembledTrue[i]);
  // LCOV_EXCL_STOP

  // Cleanup
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_diff);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_diff);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedElemRestrictionDestroy(&Erestrictqi);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
