/// @file
/// Test creation, transpose use, and destruction of an interlaced multi-component element restriction
/// \test Test creation, transpose use, and destruction of an interlaced multi-component element restriction
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 5;
  CeedInt             ind[2 * num_elem];
  CeedInt             e_layout[3];
  CeedScalar          mult;
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);
  CeedVectorCreate(ceed, 2 * (num_elem + 1), &y);
  CeedVectorSetValue(y, 0.0);  // Allocates array

  // Setup
  CeedVectorCreate(ceed, 2 * (num_elem * 2), &x);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = 2 * i;
    ind[2 * i + 1] = 2 * (i + 1);
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 2, 1, 2 * (num_elem + 1), CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);

  // Set x data in backend E-layout
  CeedElemRestrictionGetELayout(elem_restriction, e_layout);
  {
    CeedScalar x_array[2 * (num_elem * 2)];

    for (CeedInt i = 0; i < 2; i++) {             // Node
      for (CeedInt j = 0; j < 2; j++) {           // Component
        for (CeedInt k = 0; k < num_elem; k++) {  // Element
          x_array[i * e_layout[0] + j * e_layout[1] + k * e_layout[2]] = 10 * j + (2 * k + i + 1) / 2;
        }
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }

  // Restrict
  CeedElemRestrictionApply(elem_restriction, CEED_TRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  // Check
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    for (CeedInt i = 0; i < num_elem + 1; i++) {
      mult = i > 0 && i < num_elem ? 2 : 1;
      if (y_array[2 * i] != i * mult) {
        // LCOV_EXCL_START
        printf("Error in restricted array y[%" CeedInt_FMT "] = %f != %f\n", 2 * i, (CeedScalar)y_array[2 * i], i * mult);
        // LCOV_EXCL_STOP
      }
      if (y_array[2 * i + 1] != (10 + i) * mult) {
        // LCOV_EXCL_START
        printf("Error in restricted array y[%" CeedInt_FMT "] = %f != %f\n", 2 * i + 1, (CeedScalar)y_array[2 * i + 1], (10. + i) * mult);
        // LCOV_EXCL_STOP
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
