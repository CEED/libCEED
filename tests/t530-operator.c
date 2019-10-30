/// @file
/// Test assembly of mass matrix operator QFunction
/// \test Test assembly of mass matrix operator QFunction
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t510-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictxi, Erestrictui, Erestrictlini;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, A, u, v;
  const CeedScalar *a, *q;
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
  CeedElemRestrictionCreate(ceed, nelem, P*P, ndofs, dim, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);
  CeedElemRestrictionCreateIdentity(ceed, nelem, P*P, nelem*P*P, dim,
                                    &Erestrictxi);

  CeedElemRestrictionCreate(ceed, nelem, P*P, ndofs, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictu);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q, nqpts, 1, &Erestrictui);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &bu);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "_weight", Erestrictxi, CEED_NOTRANSPOSE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE, bx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "u", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", Erestrictu, CEED_NOTRANSPOSE, bu,
                       CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  // Assemble QFunction
  CeedOperatorAssembleLinearQFunction(op_mass, &A, &Erestrictlini,
                                      CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  CeedVectorGetArrayRead(qdata, CEED_MEM_HOST, &q);
  for (CeedInt i=0; i<nqpts; i++)
    if (fabs(q[i] - a[i]) > 1e-9)
      // LCOV_EXCL_START
      printf("Error: A[%d] = %f != %f\n", i, a[i], q[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(A, &a);
  CeedVectorRestoreArrayRead(qdata, &q);

  // Apply original Mass Operator
  CeedVectorCreate(ceed, ndofs, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, ndofs, &v);
  CeedVectorSetValue(v, 0.0);
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

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

  // Switch to new qdata
  CeedVectorGetArrayRead(A, CEED_MEM_HOST, &a);
  CeedVectorSetArray(qdata, CEED_MEM_HOST, CEED_COPY_VALUES, (CeedScalar *)a);
  CeedVectorRestoreArrayRead(A, &a);

  // Apply new Mass Operator
  CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);

  // Check output
  area = 0.0;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &vv);
  for (CeedInt i=0; i<ndofs; i++)
    area += vv[i];
  CeedVectorRestoreArrayRead(v, &vv);
  if (fabs(area - 1.0) > 1e-10)
    // LCOV_EXCL_START
    printf("Error: Linearized operator computed area = %f != 1.0\n", area);
  // LCOV_EXCL_STOP

  // Cleanup
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedElemRestrictionDestroy(&Erestrictlini);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&A);
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
