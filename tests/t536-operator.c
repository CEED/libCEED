/// @file
/// Test assembly of mass and Poisson operator QFunction
/// \test Test assembly of mass and Poisson operator QFunction
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t320-basis.h"
#include "t535-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictxi, Erestrictui,
                      Erestrictqi;
  CeedBasis bx, bu;
  CeedQFunction qf_setup_mass, qf_setup_diff, qf_apply;
  CeedOperator op_setup_mass, op_setup_diff, op_apply;
  CeedVector qdata_mass, qdata_diff, X, A, U, V;
  CeedInt nelem = 12, dim = 2, P = 6, Q = 4;
  CeedInt nx = 3, ny = 2;
  CeedInt row, col, offset;
  CeedInt ndofs = (nx*2+1)*(ny*2+1), nqpts = nelem*Q*Q;
  CeedInt indx[nelem*P*P];
  CeedScalar x[dim*ndofs], assembledTrue[ndofs];
  CeedScalar qref[dim*Q], qweight[Q];
  CeedScalar interp[P*Q], grad[dim*P*Q];
  CeedScalar *u;
  const CeedScalar *a, *v;

  CeedInit(argv[1], &ceed);

  // DoF Coordinates
  for (CeedInt i=0; i<ndofs; i++) {
    x[i] = (1. / (nx*2)) * (CeedScalar) (i % (nx*2+1));
    x[i+ndofs] = (1. / (ny*2)) * (CeedScalar) (i / (nx*2+1));
  }
  CeedVectorCreate(ceed, dim*ndofs, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // Qdata Vectors
  CeedVectorCreate(ceed, nqpts, &qdata_mass);
  CeedVectorCreate(ceed, nqpts*dim*(dim+1)/2, &qdata_diff);

  // Element Setup
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
  CeedElemRestrictionCreate(ceed, nelem, P, ndofs, dim, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);
  CeedElemRestrictionCreateIdentity(ceed, nelem, P, nelem*P, dim, &Erestrictxi);

  CeedElemRestrictionCreate(ceed, nelem, P, ndofs, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictu);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q, nqpts, 1, &Erestrictui);


  CeedElemRestrictionCreateIdentity(ceed, nelem, Q, nqpts, dim*(dim+1)/2,
                                    &Erestrictqi);

  // Bases
  buildmats(qref, qweight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TRIANGLE, dim, P, Q, interp, grad, qref,
                    qweight, &bx);

  buildmats(qref, qweight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TRIANGLE, 1, P, Q, interp, grad, qref,
                    qweight, &bu);

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

  // Assemble diagonal
  CeedOperatorAssembleLinearDiagonal(op_apply, &A, CEED_REQUEST_IMMEDIATE);

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
    CeedOperatorApply(op_apply, U, V, CEED_REQUEST_IMMEDIATE);

    // Retrieve entry
    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    assembledTrue[i] = v[i];
    CeedVectorRestoreArrayRead(V, &v);
  }

  // Check output
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  for (int i=0; i<ndofs; i++)
    if (fabs(a[i] - assembledTrue[i]) > 1E-14)
      // LCOV_EXCL_START
      printf("[%d] Error in assembly: %f != %f\n", i, a[i], assembledTrue[i]);
  // LCOV_EXCL_STOP

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedElemRestrictionDestroy(&Erestrictqi);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&qdata_mass);
  CeedVectorDestroy(&qdata_diff);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
