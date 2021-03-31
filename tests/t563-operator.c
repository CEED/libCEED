/// @file
/// Test full assembly of mass and Poisson operator (see t536)
/// \test Test full assembly of mass and Poisson operator
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t320-basis.h"
#include "t535-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictui, Erestrictqi;
  CeedBasis bx, bu;
  CeedQFunction qf_setup_mass, qf_setup_diff, qf_apply;
  CeedOperator op_setup_mass, op_setup_diff, op_apply;
  CeedVector qdata_mass, qdata_diff, X, U, V;
  CeedInt nelem = 12, dim = 2, P = 6, Q = 4;
  CeedInt nx = 3, ny = 2;
  CeedInt row, col, offset;
  CeedInt ndofs = (nx*2+1)*(ny*2+1), nqpts = nelem*Q;
  CeedInt indx[nelem*P*P];
  CeedScalar assembled[ndofs*ndofs];
  CeedScalar x[dim*ndofs], assembledTrue[ndofs*ndofs];
  CeedScalar qref[dim*Q], qweight[Q];
  CeedScalar interp[P*Q], grad[dim*P*Q];
  CeedScalar *u;
  const CeedScalar *v;

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
  CeedElemRestrictionCreate(ceed, nelem, P, dim, ndofs, dim*ndofs,
                            CEED_MEM_HOST, CEED_USE_POINTER, indx, &Erestrictx);

  CeedElemRestrictionCreate(ceed, nelem, P, 1, 1, ndofs, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictu);
  CeedInt stridesu[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q, 1, nqpts, stridesu,
                                   &Erestrictui);

  CeedInt stridesqd[3] = {1, Q, Q *dim *(dim+1)/2};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q, dim*(dim+1)/2,
                                   dim*(dim+1)/2*nqpts,
                                   stridesqd, &Erestrictqi);

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
  CeedOperatorCreate(ceed, qf_setup_mass, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setup_mass);
  CeedOperatorSetField(op_setup_mass, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_mass, "_weight", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_mass, "qdata", Erestrictui,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diff
  CeedQFunctionCreateInterior(ceed, 1, setup_diff, setup_diff_loc,
                              &qf_setup_diff);
  CeedQFunctionAddInput(qf_setup_diff, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_diff, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_diff, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);

  // Operator - setup diff
  CeedOperatorCreate(ceed, qf_setup_diff, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setup_diff);
  CeedOperatorSetField(op_setup_diff, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_diff, "_weight", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_diff, "qdata", Erestrictqi,
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
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_apply);
  CeedOperatorSetField(op_apply, "du", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata_mass", Erestrictui,
                       CEED_BASIS_COLLOCATED, qdata_mass);
  CeedOperatorSetField(op_apply, "qdata_diff", Erestrictqi,
                       CEED_BASIS_COLLOCATED, qdata_diff);
  CeedOperatorSetField(op_apply, "u", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "v", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "dv", Erestrictu, bu, CEED_VECTOR_ACTIVE);

  // Fully assemble operator
  for (int k=0; k<ndofs*ndofs; ++k) {
    assembled[k] = 0.0;
    assembledTrue[k] = 0.0;
  }
  CeedInt nentries;
  CeedInt *rows;
  CeedInt *cols;
  CeedVector values;
  CeedOperatorLinearAssembleSymbolic(op_apply, &nentries, &rows, &cols);
  CeedVectorCreate(ceed, nentries, &values);
  CeedOperatorLinearAssemble(op_apply, values);
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
    CeedOperatorApply(op_apply, U, V, CEED_REQUEST_IMMEDIATE);

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
  CeedQFunctionDestroy(&qf_setup_mass);
  CeedQFunctionDestroy(&qf_setup_diff);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_setup_mass);
  CeedOperatorDestroy(&op_setup_diff);
  CeedOperatorDestroy(&op_apply);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictqi);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&qdata_mass);
  CeedVectorDestroy(&qdata_diff);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
