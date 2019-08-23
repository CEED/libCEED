/// @file
/// Test calculation of dof multiplicity in element restriction
/// \test Test calculation of dof multiplicity in element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector mult;
  CeedInt ne = 3;
  CeedInt ind[4*ne];
  const CeedScalar *mm;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 3*ne+1, &mult);
  CeedVectorSetValue(mult, 0); // Allocates array

  for (CeedInt i=0; i<ne; i++) {
    ind[4*i+0] = i*3+0;
    ind[4*i+1] = i*3+1;
    ind[4*i+2] = i*3+2;
    ind[4*i+3] = i*3+3;
  }
  CeedElemRestrictionCreate(ceed, ne, 4, 3*ne+1, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, ind, &r);

  CeedElemRestrictionGetMultiplicity(r, mult);

  CeedVectorGetArrayRead(mult, CEED_MEM_HOST, &mm);
  for (CeedInt i=0; i<3*ne+1; i++)
    if ((1 + (i > 0 && i < 3*ne && (i%3==0) ? 1 : 0)) != mm[i])
      // LCOV_EXCL_START
      printf("Error in multiplicity vector: mult[%d] = %f\n",
             i, (double)mm[i]);
      // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(mult, &mm);

  CeedVectorDestroy(&mult);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
