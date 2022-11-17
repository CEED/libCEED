/// @file
/// Test creation, transpose use, and destruction of a multicomponent element restriction
/// \test Test creation, transpose use, and destruction of a multicomponent element restriction
#include <ceed.h>
#include <ceed/backend.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 5;
  CeedInt             ind[2 * num_elem];
  CeedInt             layout[3];
  CeedScalar          mult;
  CeedScalar          a[2 * (num_elem * 2)];
  const CeedScalar   *yy;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  // Setup
  CeedVectorCreate(ceed, 2 * (num_elem * 2), &x);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 2, num_elem + 1, 2 * (num_elem + 1), CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, 2 * (num_elem + 1), &y);
  CeedVectorSetValue(y, 0);  // Allocates array

  // Set x data in backend E-layout
  CeedElemRestrictionGetELayout(r, &layout);
  for (CeedInt i = 0; i < 2; i++) {             // Node
    for (CeedInt j = 0; j < 2; j++) {           // Component
      for (CeedInt k = 0; k < num_elem; k++) {  // Element
        a[i * layout[0] + j * layout[1] + k * layout[2]] = 10 * j + (2 * k + i + 1) / 2;
      }
    }
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  // Restrict
  CeedElemRestrictionApply(r, CEED_TRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  // Check
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  for (CeedInt i = 0; i < num_elem + 1; i++) {
    mult = i > 0 && i < num_elem ? 2 : 1;
    if (yy[i] != i * mult) printf("Error in restricted array y[%" CeedInt_FMT "] = %f != %f\n", i, (CeedScalar)yy[i], i * mult);
    if (yy[i + num_elem + 1] != (10 + i) * mult) {
      // LCOV_EXCL_START
      printf("Error in restricted array y[%" CeedInt_FMT "] = %f != %f\n", i + num_elem + 1, (CeedScalar)yy[i + num_elem + 1], (10. + i) * mult);
      // LCOV_EXCL_STOP
    }
  }

  CeedVectorRestoreArrayRead(y, &yy);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
