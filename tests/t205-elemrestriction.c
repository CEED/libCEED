/// @file
/// Test creation, use, and destruction of an interlaced multi-component element restriction
/// \test Test creation, use, and destruction of an interlaced multi-component element restriction
#include <ceed.h>
#include <ceed/backend.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 3;
  CeedInt             ind[2 * num_elem];
  CeedInt             layout[3];
  CeedScalar          x_array[2 * (num_elem + 1)];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  // Setup
  CeedVectorCreate(ceed, 2 * (num_elem + 1), &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) {
    x_array[2 * i]     = 10 + i;
    x_array[2 * i + 1] = 20 + i;
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, 2 * (num_elem * 2), &y);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = 2 * i;
    ind[2 * i + 1] = 2 * (i + 1);
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 2, 1, 2 * (num_elem + 1), CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);

  // Restrict
  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  // Check
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    CeedElemRestrictionGetELayout(elem_restriction, &layout);
    for (CeedInt i = 0; i < 2; i++) {             // Node
      for (CeedInt j = 0; j < 2; j++) {           // Component
        for (CeedInt k = 0; k < num_elem; k++) {  // Element
          if (y_array[i * layout[0] + j * layout[1] + k * layout[2]] != x_array[ind[i + k * 2] + j]) {
            // LCOV_EXCL_START
            printf("Error in restricted array y[%" CeedInt_FMT "][%" CeedInt_FMT "][%" CeedInt_FMT "] = %f != %f\n", i, j, k,
                   (CeedScalar)y_array[i * layout[0] + j * layout[1] + k * layout[2]], x_array[ind[i + k * 2] + j * (num_elem + 1)]);
            // LCOV_EXCL_STOP
          }
        }
      }
    }
    CeedVectorRestoreArrayRead(y, &y_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
