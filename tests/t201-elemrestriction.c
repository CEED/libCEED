/// @file
/// Test creation, use, and destruction of a strided element restriction
/// \test Test creation, use, and destruction of a strided element restriction
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem   = 3;
  CeedInt             strides[3] = {1, 2, 2};
  CeedInt             layout[3];
  CeedScalar          x_array[num_elem * 2];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem * 2, &x);
  for (CeedInt i = 0; i < num_elem * 2; i++) x_array[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, num_elem * 2, &y);

  CeedElemRestrictionCreateStrided(ceed, num_elem, 2, 1, num_elem * 2, strides, &elem_restriction);
  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    CeedElemRestrictionGetELayout(elem_restriction, &layout);
    for (CeedInt i = 0; i < 2; i++) {             // Node
      for (CeedInt j = 0; j < 1; j++) {           // Component
        for (CeedInt k = 0; k < num_elem; k++) {  // Element
          if (y_array[i * layout[0] + j * layout[1] + k * layout[2]] != x_array[i * strides[0] + j * strides[1] + k * strides[2]]) {
            // LCOV_EXCL_START
            printf("Error in restricted array y[%" CeedInt_FMT "][%" CeedInt_FMT "][%" CeedInt_FMT "] = %f\n", i, j, k,
                   (CeedScalar)y_array[i * strides[0] + j * strides[1] + j * strides[2]]);
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
