/// @file
/// Test full assembly of composite operator (see t538)
/// \test Test full assembly of composite operator
#include <ceed.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu,
                      Erestrictui, ErestrictqiMass, ErestrictqiDiff;
  CeedBasis bx, bu;
  CeedQFunction qf_setupMass, qf_mass, qf_setupDiff, qf_diff;
  CeedOperator op_setupMass, op_mass, op_setupDiff, op_diff, op_apply;
  CeedVector qdataMass, qdataDiff, X, U, V;
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

  // Qdata Vectors
  CeedVectorCreate(ceed, nqpts, &qdataMass);
  CeedVectorCreate(ceed, nqpts*dim*(dim+1)/2, &qdataDiff);

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

  CeedInt stridesqdMass[3] = {1, Q*Q, Q*Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, 1, nqpts,
                                   stridesqdMass, &ErestrictqiMass);
  CeedInt stridesqdDiff[3] = {1, Q*Q, Q*Q*dim*(dim+1)/2}; /* *NOPAD* */
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, dim*(dim+1)/2,
                                   dim*(dim+1)/2*nqpts,
                                   stridesqdDiff, &ErestrictqiDiff);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, P, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P, Q, CEED_GAUSS, &bu);

  // QFunction - setup mass
  CeedQFunctionCreateInteriorByName(ceed, "Mass2DBuild", &qf_setupMass);

  // Operator - setup mass
  CeedOperatorCreate(ceed, qf_setupMass, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setupMass);
  CeedOperatorSetField(op_setupMass, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupMass, "weights", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupMass, "qdata", ErestrictqiMass,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // QFunction - setup diffusion
  CeedQFunctionCreateInteriorByName(ceed, "Poisson2DBuild", &qf_setupDiff);

  // Operator - setup diffusion
  CeedOperatorCreate(ceed, qf_setupDiff, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setupDiff);
  CeedOperatorSetField(op_setupDiff, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupDiff, "weights", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupDiff, "qdata", ErestrictqiDiff,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Apply Setup Operators
  CeedOperatorApply(op_setupMass, X, qdataMass, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setupDiff, X, qdataDiff, CEED_REQUEST_IMMEDIATE);

  // QFunction - apply mass
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  // Operator - apply mass
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_mass);
  CeedOperatorSetField(op_mass, "u", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", ErestrictqiMass, CEED_BASIS_COLLOCATED,
                       qdataMass);
  CeedOperatorSetField(op_mass, "v", Erestrictu, bu, CEED_VECTOR_ACTIVE);

  // QFunction - apply diff
  CeedQFunctionCreateInteriorByName(ceed, "Poisson2DApply", &qf_diff);

  // Operator - apply
  CeedOperatorCreate(ceed, qf_diff, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_diff);
  CeedOperatorSetField(op_diff, "du", Erestrictu, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diff, "qdata", ErestrictqiDiff, CEED_BASIS_COLLOCATED,
                       qdataDiff);
  CeedOperatorSetField(op_diff, "dv", Erestrictu, bu, CEED_VECTOR_ACTIVE);

  // Composite operator
  CeedCompositeOperatorCreate(ceed, &op_apply);
  CeedCompositeOperatorAddSub(op_apply, op_mass);
  CeedCompositeOperatorAddSub(op_apply, op_diff);

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
  CeedQFunctionDestroy(&qf_setupMass);
  CeedQFunctionDestroy(&qf_setupDiff);
  CeedQFunctionDestroy(&qf_diff);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setupMass);
  CeedOperatorDestroy(&op_setupDiff);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_diff);
  CeedOperatorDestroy(&op_apply);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&ErestrictqiMass);
  CeedElemRestrictionDestroy(&ErestrictqiDiff);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&qdataMass);
  CeedVectorDestroy(&qdataDiff);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
