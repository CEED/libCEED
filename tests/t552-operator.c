/// @file
/// Test creation, action, and destruction for mass matrix operator with multigrid level, nontensor basis
/// \test Test creation, action, and destruction for mass matrix operator with multigrid level, nontensor basis
#include <ceed.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictui,
                      ErestrictuCoarse, ErestrictuFine;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_massCoarse, op_massFine,
               op_prolong, op_restrict;
  CeedVector qdata, X, Ucoarse, Ufine,
             Vcoarse, Vfine, PMultFine;
  const CeedScalar *hv;
  CeedInt nelem = 15, Pcoarse = 3, Pfine = 5, Q = 8;
  CeedInt Nx = nelem+1, NuCoarse = nelem*(Pcoarse-1)+1,
          NuFine = nelem*(Pfine-1)+1;
  CeedInt induCoarse[nelem*Pcoarse], induFine[nelem*Pfine],
          indx[nelem*2];
  CeedScalar x[Nx];
  CeedScalar sum;

  CeedInit(argv[1], &ceed);

  for (CeedInt i=0; i<Nx; i++)
    x[i] = (CeedScalar) i / (Nx - 1);
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
  // Restrictions
  CeedElemRestrictionCreate(ceed, nelem, 2, 1, 1, Nx, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<Pcoarse; j++) {
      induCoarse[Pcoarse*i+j] = i*(Pcoarse-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, Pcoarse, 1, 1, NuCoarse, CEED_MEM_HOST,
                            CEED_USE_POINTER, induCoarse, &ErestrictuCoarse);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<Pfine; j++) {
      induFine[Pfine*i+j] = i*(Pfine-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, Pfine, 1, 1, NuFine, CEED_MEM_HOST,
                            CEED_USE_POINTER, induFine, &ErestrictuFine);

  CeedInt stridesu[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q, 1, Q*nelem, stridesu,
                                   &Erestrictui);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Pfine, Q, CEED_GAUSS, &bu);

  // QFunctions
  CeedQFunctionCreateInteriorByName(ceed, "Mass1DBuild", &qf_setup);
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup);
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_massFine);

  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorCreate(ceed, nelem*Q, &qdata);

  CeedOperatorSetField(op_setup, "weights", CEED_ELEMRESTRICTION_NONE, bx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, bx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "qdata", Erestrictui, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_massFine, "qdata", Erestrictui, CEED_BASIS_COLLOCATED,
                       qdata);
  CeedOperatorSetField(op_massFine, "u", ErestrictuFine, bu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_massFine, "v", ErestrictuFine, bu, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  // Create multigrid level
  CeedVectorCreate(ceed, NuFine, &PMultFine);
  CeedVectorSetValue(PMultFine, 1.0);
  CeedBasis buCoarse, bCtoF;
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Pcoarse, Q, CEED_GAUSS, &buCoarse);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Pcoarse, Pfine, CEED_GAUSS_LOBATTO,
                                  &bCtoF);
  const CeedScalar *interpCtoF;
  CeedBasisGetInterp1D(bCtoF, &interpCtoF);
  CeedOperatorMultigridLevelCreateH1(PMultFine, ErestrictuCoarse, buCoarse,
                                     interpCtoF, op_massFine, &op_massCoarse,
                                     &op_prolong, &op_restrict);

  // Coarse problem
  CeedVectorCreate(ceed, NuCoarse, &Ucoarse);
  CeedVectorSetValue(Ucoarse, 1.0);
  CeedVectorCreate(ceed, NuCoarse, &Vcoarse);
  CeedOperatorApply(op_massCoarse, Ucoarse, Vcoarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(Vcoarse, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i=0; i<NuCoarse; i++) {
    sum += hv[i];
  }
  if (fabs(sum-1.)>1e-10)
    printf("Computed Area Coarse Grid: %f != True Area: 1.0\n", sum);
  CeedVectorRestoreArrayRead(Vcoarse, &hv);

  // Prolong coarse u
  CeedVectorCreate(ceed, NuFine, &Ufine);
  CeedOperatorApply(op_prolong, Ucoarse, Ufine, CEED_REQUEST_IMMEDIATE);

  // Fine problem
  CeedVectorCreate(ceed, NuFine, &Vfine);
  CeedOperatorApply(op_massFine, Ufine, Vfine, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(Vfine, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i=0; i<NuFine; i++) {
    sum += hv[i];
  }
  if (fabs(sum-1.)>1e-10)
    printf("Computed Area Fine Grid: %f != True Area: 1.0\n", sum);
  CeedVectorRestoreArrayRead(Vfine, &hv);

  // Restrict state to coarse grid
  CeedOperatorApply(op_restrict, Vfine, Vcoarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(Vcoarse, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i=0; i<NuCoarse; i++) {
    sum += hv[i];
  }
  if (fabs(sum-1.)>1e-10)
    printf("Computed Area Coarse Grid: %f != True Area: 1.0\n", sum);
  CeedVectorRestoreArrayRead(Vcoarse, &hv);

  // Cleanup
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_massCoarse);
  CeedOperatorDestroy(&op_massFine);
  CeedOperatorDestroy(&op_prolong);
  CeedOperatorDestroy(&op_restrict);
  CeedElemRestrictionDestroy(&ErestrictuCoarse);
  CeedElemRestrictionDestroy(&ErestrictuFine);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&buCoarse);
  CeedBasisDestroy(&bCtoF);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Ucoarse);
  CeedVectorDestroy(&Ufine);
  CeedVectorDestroy(&Vcoarse);
  CeedVectorDestroy(&Vfine);
  CeedVectorDestroy(&PMultFine);
  CeedVectorDestroy(&qdata);
  CeedDestroy(&ceed);
  return 0;
}