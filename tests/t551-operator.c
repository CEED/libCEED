/// @file
/// Test creation, action, and destruction for mass matrix operator with multigrid level, non-tensor basis and interpolation basis generation
/// \test Test creation, action, and destruction for mass matrix operator with multigrid level, non-tensor basis and interpolation basis generation
#include <ceed.h>
#include <stdlib.h>
#include <math.h>

#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictui,
                      ErestrictuCoarse, ErestrictuFine;
  CeedBasis bx, bTemp, bCoarse, bFine;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_massCoarse, op_massFine,
               op_prolong, op_restrict;
  CeedVector qdata, X, Ucoarse, Ufine,
             Vcoarse, Vfine, PMultFine;
  const CeedScalar *hv;
  CeedInt nelem = 15, Pcoarse = 3, Pfine = 5, Q = 8, ncomp = 2;
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
  CeedElemRestrictionCreate(ceed, nelem, Pcoarse, ncomp, NuCoarse,
                            ncomp*NuCoarse, CEED_MEM_HOST, CEED_USE_POINTER,
                            induCoarse, &ErestrictuCoarse);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<Pfine; j++) {
      induFine[Pfine*i+j] = i*(Pfine-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, Pfine, ncomp, NuFine,
                            ncomp*NuFine, CEED_MEM_HOST, CEED_USE_POINTER,
                            induFine, &ErestrictuFine);

  CeedInt stridesu[3] = {1, Q, Q};
  CeedElemRestrictionCreateStrided(ceed, nelem, Q, 1, Q*nelem, stridesu,
                                   &Erestrictui);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, ncomp, Pcoarse, Q, CEED_GAUSS,
                                  &bTemp);
  const CeedScalar *interp, *grad, *qref, *qweight;
  CeedBasisGetInterp1D(bTemp, &interp);
  CeedBasisGetGrad1D(bTemp, &grad);
  CeedBasisGetQRef(bTemp, &qref);
  CeedBasisGetQWeights(bTemp, &qweight);
  CeedBasisCreateH1(ceed, CEED_LINE, ncomp, Pcoarse, Q, interp, grad, qref,
                    qweight, &bCoarse);
  CeedBasisDestroy(&bTemp);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, ncomp, Pfine, Q, CEED_GAUSS, &bTemp);
  CeedBasisGetInterp1D(bTemp, &interp);
  CeedBasisGetGrad1D(bTemp, &grad);
  CeedBasisGetQRef(bTemp, &qref);
  CeedBasisGetQWeights(bTemp, &qweight);
  CeedBasisCreateH1(ceed, CEED_LINE, ncomp, Pfine, Q, interp, grad, qref,
                    qweight, &bFine);
  CeedBasisDestroy(&bTemp);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weights", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", 1*1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", ncomp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", ncomp, CEED_EVAL_INTERP);

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
  CeedOperatorSetField(op_massFine, "u", ErestrictuFine, bFine,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_massFine, "v", ErestrictuFine, bFine,
                       CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_setup, X, qdata, CEED_REQUEST_IMMEDIATE);

  // Create multigrid level
  CeedVectorCreate(ceed, ncomp*NuFine, &PMultFine);
  CeedVectorSetValue(PMultFine, 1.0);
  CeedOperatorMultigridLevelCreate(op_massFine, PMultFine, ErestrictuCoarse,
                                   bCoarse, &op_massCoarse, &op_prolong, &op_restrict);

  // Coarse problem
  CeedVectorCreate(ceed, ncomp*NuCoarse, &Ucoarse);
  CeedVectorSetValue(Ucoarse, 1.0);
  CeedVectorCreate(ceed, ncomp*NuCoarse, &Vcoarse);
  CeedOperatorApply(op_massCoarse, Ucoarse, Vcoarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(Vcoarse, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i=0; i<ncomp*NuCoarse; i++) {
    sum += hv[i];
  }
  if (fabs(sum-2.)>1e-10)
    // LCOV_EXCL_START
    printf("Computed Area Coarse Grid: %f != True Area: 1.0\n", sum);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(Vcoarse, &hv);

  // Prolong coarse u
  CeedVectorCreate(ceed, ncomp*NuFine, &Ufine);
  CeedOperatorApply(op_prolong, Ucoarse, Ufine, CEED_REQUEST_IMMEDIATE);

  // Fine problem
  CeedVectorCreate(ceed, ncomp*NuFine, &Vfine);
  CeedOperatorApply(op_massFine, Ufine, Vfine, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(Vfine, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i=0; i<ncomp*NuFine; i++) {
    sum += hv[i];
  }
  if (fabs(sum-2.)>1e-10)
    // LCOV_EXCL_START
    printf("Computed Area Fine Grid: %f != True Area: 1.0\n", sum);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(Vfine, &hv);

  // Restrict state to coarse grid
  CeedOperatorApply(op_restrict, Vfine, Vcoarse, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(Vcoarse, CEED_MEM_HOST, &hv);
  sum = 0.;
  for (CeedInt i=0; i<ncomp*NuCoarse; i++) {
    sum += hv[i];
  }
  if (fabs(sum-2.)>1e-10)
    // LCOV_EXCL_START
    printf("Computed Area Coarse Grid: %f != True Area: 1.0\n", sum);
  // LCOV_EXCL_STOP
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
  CeedBasisDestroy(&bCoarse);
  CeedBasisDestroy(&bFine);
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
