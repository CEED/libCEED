/// @file
/// Test full assembly of Poisson operator
/// \test Test full assembly of Poisson operator
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t534-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictui, Erestrictqi;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_diff;
  CeedOperator op_setup, op_diff;
  CeedVector qdata, X, U, V;
  CeedInt P = 3, Q = 4, dim = 2;
  CeedInt nx = 3, ny = 2;
  CeedInt nelem = nx * ny;
  CeedInt ndofs = (nx*2+1)*(ny*2+1), nqpts = nelem*Q*Q;
  CeedInt indx[nelem*P*P];
  CeedScalar assembled[ndofs*ndofs];
  CeedScalar x[dim*ndofs], assembledTrue[ndofs*ndofs];
  CeedScalar *u;
  const CeedScalar *v;

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
  CeedElemRestrictionCreate(ceed, nelem, P*P, dim, ndofs, dim*ndofs,
                            CEED_MEM_HOST, CEED_USE_POINTER, indx, &Erestrictx);

  CeedElemRestrictionCreate(ceed, nelem, P*P, 1, 1, ndofs, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictu);
  CeedInt stridesu[3] = {1, Q*Q, Q*Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, 1, nqpts, stridesu,
                                   &Erestrictui);

  CeedInt stridesqd[3] = {1, Q*Q, Q *Q *dim *(dim+1)/2};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, dim*(dim+1)/2,
                                   dim*(dim+1)/2*nqpts,
                                   stridesqd, &Erestrictqi);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &bu);

  // QFunction - setup
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);

  // Operator - setup
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "_weight", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", Erestrictqi, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply
  CeedQFunctionCreateInterior(ceed, 1, diff, diff_loc, &qf_diff);
  CeedQFunctionAddInput(qf_diff, "du", dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diff, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diff, "dv", dim, CEED_EVAL_GRAD);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_diff);
  CeedOperatorSetField(op_diff, "du", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "qdata", Erestrictqi, CEED_BASIS_COLLOCATED,
                       qdata);
  CeedOperatorSetField(op_diff, "dv", Erestrictu, bu, CEED_VECTOR_ACTIVE);

  // Fully assemble operator
  for (int k=0; k<ndofs*ndofs; ++k) {
    assembled[k] = 0.0;
    assembledTrue[k] = 0.0;
  }
  CeedInt nentries;
  CeedInt *rows;
  CeedInt *cols;
  CeedVector values;
  CeedOperatorLinearAssembleSymbolic(op_diff, &nentries, &rows, &cols);
  CeedVectorCreate(ceed, nentries, &values);
  CeedOperatorLinearAssemble(op_diff, values);
  const CeedScalar *vals;
  CeedVectorGetArrayRead(values, CEED_MEM_HOST, &vals);
  for (int k=0; k<nentries; ++k) {
    assembled[rows[k]*ndofs + cols[k]] += vals[k];
  }
  CeedVectorRestoreArrayRead(values, &vals);

  // Manually assemble operator
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

    // Compute entries for column i
    CeedOperatorApply(op_diff, U, V, CEED_REQUEST_IMMEDIATE);

    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    for (int k=0; k<ndofs; k++) {
      assembledTrue[i*ndofs + k] = v[k];
    }
    CeedVectorRestoreArrayRead(V, &v);
  }

  // Check output
  for (int i=0; i<ndofs; i++)
    for (int j=0; j<ndofs; j++)
      if (fabs(assembled[j*ndofs+i] - assembledTrue[j*ndofs+i]) > 1e-14)
        // LCOV_EXCL_START
        printf("[%d,%d] Error in assembly: %f != %f\n", i, j,
               assembled[j*ndofs+i], assembledTrue[j*ndofs+i]);
  // LCOV_EXCL_STOP

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&values);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_diff);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_diff);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictqi);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
