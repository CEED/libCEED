/// @file
/// Test calculation of DoF multiplicity in element restriction
/// \test Test calculation of DoF multiplicity in element restriction
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          mult;
  CeedInt             num_elem = 3;
  CeedInt             ind[4 * num_elem];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 3 * num_elem + 1, &mult);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[4 * i + 0] = i * 3 + 0;
    ind[4 * i + 1] = i * 3 + 1;
    ind[4 * i + 2] = i * 3 + 2;
    ind[4 * i + 3] = i * 3 + 3;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 4, 1, 1, 3 * num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);

  CeedElemRestrictionGetMultiplicity(elem_restriction, mult);
  {
    const CeedScalar *mult_array;

    CeedVectorGetArrayRead(mult, CEED_MEM_HOST, &mult_array);
    for (CeedInt i = 0; i < 3 * num_elem + 1; i++) {
      if (mult_array[i] != (1 + (i > 0 && i < 3 * num_elem && (i % 3 == 0) ? 1 : 0))) {
        // LCOV_EXCL_START
        printf("Error in multiplicity vector: mult[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)mult_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(mult, &mult_array);
  }

  CeedVectorDestroy(&mult);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
