/// @file
/// Test assembly of mass matrix operator (multi-component) see t537
/// \test Test assembly of mass matrix operator (multi-component)
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t537-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictui;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, U, V;
  CeedInt P = 3, Q = 4, dim = 2, ncomp = 2;
  // CeedInt nx = 3, ny = 2;
  CeedInt nx = 1, ny = 1;
  CeedInt nelem = nx * ny;
  CeedInt ndofs = (nx*2+1)*(ny*2+1), nqpts = nelem*Q*Q;
  CeedInt indx[nelem*P*P];
  CeedScalar assembled[ncomp*ncomp*ndofs*ndofs];
  CeedScalar x[dim*ndofs], assembledTrue[ncomp*ncomp*ndofs*ndofs];
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
  CeedVectorCreate(ceed, nqpts, &qdata);

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
  CeedElemRestrictionCreate(ceed, nelem, P*P, ncomp, ndofs, ncomp*ndofs,
                            CEED_MEM_HOST, CEED_USE_POINTER, indx, &Erestrictu);
  CeedInt stridesu[3] = {1, Q*Q, Q*Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, 1, nqpts, stridesu,
                                   &Erestrictui);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncomp, P, Q, CEED_GAUSS, &bu);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", ncomp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", ncomp, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup);
  CeedOperatorSetField(op_setup, "_weight", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", Erestrictui, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_mass);
  CeedOperatorSetField(op_mass, "rho", Erestrictui, CEED_BASIS_COLLOCATED,
                       qdata);
  CeedOperatorSetField(op_mass, "u", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", Erestrictu, bu, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  // Fuly assemble operator
  for (int k=0; k<ncomp*ncomp*ndofs*ndofs; ++k) {
    assembled[k] = 0.0;
    assembledTrue[k] = 0.0;
  }
  CeedInt nentries;
  CeedInt *rows;
  CeedInt *cols;
  CeedVector values;
  CeedOperatorLinearAssembleSymbolic(op_mass, &nentries, &rows, &cols);
  CeedVectorCreate(ceed, nentries, &values);
  CeedOperatorLinearAssemble(op_mass, values);
  const CeedScalar *vals;
  CeedVectorGetArrayRead(values, CEED_MEM_HOST, &vals);
  for (int k=0; k<nentries; ++k) {
    assembled[rows[k]*ncomp*ndofs + cols[k]] += vals[k];
  }
  CeedVectorRestoreArrayRead(values, &vals);

  // Manually assemble operator
  CeedVectorCreate(ceed, ncomp*ndofs, &U);
  CeedVectorSetValue(U, 0.0);
  CeedVectorCreate(ceed, ncomp*ndofs, &V);
  CeedInt indOld = -1;
  for (int j=0; j<ndofs*ncomp; j++) {
    // Set input
    CeedVectorGetArray(U, CEED_MEM_HOST, &u);
    CeedInt ind = j;
    u[ind] = 1.0;
    if (ind > 0)
      u[indOld] = 0.0;
    indOld = ind;
    CeedVectorRestoreArray(U, &u);

    // Compute effect of DoF j
    CeedOperatorApply(op_mass, U, V, CEED_REQUEST_IMMEDIATE);

    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
    for (int k=0; k<ndofs*ncomp; k++) {
      assembledTrue[j*ndofs*ncomp + k] = v[k];
    }
    CeedVectorRestoreArrayRead(V, &v);
  }

  // Check output
  for (int i=0; i<ncomp*ndofs; i++)
    for (int j=0; j<ncomp*ndofs; j++)
      if (fabs(assembled[j*ndofs*ncomp+i] - assembledTrue[j*ndofs*ncomp+i]) > 1e-14)
        // LCOV_EXCL_START
        printf("[%d,%d] Error in assembly: %f != %f\n", i, j,
               assembled[j*ndofs*ncomp+i], assembledTrue[j*ndofs*ncomp+i]);
  // LCOV_EXCL_STOP

  // Cleanup
  free(rows);
  free(cols);
  CeedVectorDestroy(&values);
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
  // CeedVectorDestroy(&A);
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
