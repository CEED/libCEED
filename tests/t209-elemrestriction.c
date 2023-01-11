/// @file
/// Test calculation of dof multiplicity in element restriction
/// \test Test calculation of dof multiplicity in element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          mult;
  CeedInt             num_elem = 3;
  CeedInt             ind[4 * num_elem];
  const CeedScalar   *mm;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 3 * num_elem + 1, &mult);
  CeedVectorSetValue(mult, 0);  // Allocates array

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[4 * i + 0] = i * 3 + 0;
    ind[4 * i + 1] = i * 3 + 1;
    ind[4 * i + 2] = i * 3 + 2;
    ind[4 * i + 3] = i * 3 + 3;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 4, 1, 1, 3 * num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);

  CeedElemRestrictionGetMultiplicity(r, mult);

  CeedVectorGetArrayRead(mult, CEED_MEM_HOST, &mm);
  for (CeedInt i = 0; i < 3 * num_elem + 1; i++) {
    if ((1 + (i > 0 && i < 3 * num_elem && (i % 3 == 0) ? 1 : 0)) != mm[i]) {
      // LCOV_EXCL_START
      printf("Error in multiplicity vector: mult[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)mm[i]);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(mult, &mm);

  CeedVectorDestroy(&mult);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
