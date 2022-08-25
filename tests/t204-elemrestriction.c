/// @file
/// Test creation, use, and destruction of a multicomponent element restriction
/// \test Test creation, use, and destruction of a multicomponent element restriction
#include <ceed.h>
#include <ceed/backend.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 3;
  CeedInt             ind[2 * num_elem];
  CeedInt             layout[3];
  CeedScalar          a[2 * (num_elem + 1)];
  const CeedScalar   *yy;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  // Setup
  CeedVectorCreate(ceed, 2 * (num_elem + 1), &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) {
    a[i]                = 10 + i;
    a[i + num_elem + 1] = 20 + i;
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 2, num_elem + 1, 2 * (num_elem + 1), CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, 2 * (num_elem * 2), &y);

  // Restrict
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  // Check
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  CeedElemRestrictionGetELayout(r, &layout);
  for (CeedInt i = 0; i < 2; i++) {             // Node
    for (CeedInt j = 0; j < 2; j++) {           // Component
      for (CeedInt k = 0; k < num_elem; k++) {  // Element
        if (yy[i * layout[0] + j * layout[1] + k * layout[2]] != a[ind[i + k * 2] + j * (num_elem + 1)]) {
          // LCOV_EXCL_START
          printf("Error in restricted array y[%" CeedInt_FMT "][%" CeedInt_FMT "][%" CeedInt_FMT "] = %f != %f\n", i, j, k,
                 (CeedScalar)yy[i * layout[0] + j * layout[1] + k * layout[2]], a[ind[i + k * 2] + j * (num_elem + 1)]);
          // LCOV_EXCL_STOP
        }
      }
    }
  }

  CeedVectorRestoreArrayRead(y, &yy);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
