/// @file
/// Test creation, transpose use, and destruction of a multi-component element restriction
/// \test Test creation, transpose use, and destruction of a multi-component element restriction
#include <ceed.h>
#include <ceed/backend.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 5;
  CeedInt             ind[2 * num_elem];
  CeedInt             layout[3];
  CeedScalar          mult;
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  // Setup
  CeedVectorCreate(ceed, 2 * (num_elem * 2), &x);
  CeedVectorCreate(ceed, 2 * (num_elem + 1), &y);
  CeedVectorSetValue(y, 0.0);  // Allocates array, transpose mode sums into vector

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 2, num_elem + 1, 2 * (num_elem + 1), CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);

  // Set x data in backend E-layout
  CeedElemRestrictionGetELayout(elem_restriction, &layout);
  {
    CeedScalar x_array[2 * (num_elem * 2)];

    for (CeedInt i = 0; i < 2; i++) {             // Node
      for (CeedInt j = 0; j < 2; j++) {           // Component
        for (CeedInt k = 0; k < num_elem; k++) {  // Element
          x_array[i * layout[0] + j * layout[1] + k * layout[2]] = 10 * j + (2 * k + i + 1) / 2;
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
      if (y_array[i] != i * mult) printf("Error in restricted array y[%" CeedInt_FMT "] = %f != %f\n", i, (CeedScalar)y_array[i], i * mult);
      if (y_array[i + num_elem + 1] != (10 + i) * mult) {
        // LCOV_EXCL_START
        printf("Error in restricted array y[%" CeedInt_FMT "] = %f != %f\n", i + num_elem + 1, (CeedScalar)y_array[i + num_elem + 1],
               (10. + i) * mult);
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
