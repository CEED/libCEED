/// @file
/// Test creation, use, and destruction of a blocked element restriction
/// \test Test creation, use, and destruction of a blocked element restriction
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem  = 8;
  CeedInt             elem_size = 2;
  CeedInt             blk_size  = 5;
  CeedInt             ind[elem_size * num_elem];
  CeedScalar          x_array[num_elem + 1];
  CeedInt             layout[3];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem + 1, &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) x_array[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, blk_size * elem_size, &y);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, blk_size, 1, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);

  // No Transpose
  CeedElemRestrictionApplyBlock(elem_restriction, 1, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    CeedElemRestrictionGetELayout(elem_restriction, &layout);
    for (CeedInt i = 0; i < elem_size; i++) {            // Node
      for (CeedInt j = 0; j < 1; j++) {                  // Component
        for (CeedInt k = blk_size; k < num_elem; k++) {  // Element
          CeedInt block = k / blk_size;
          CeedInt elem  = k % blk_size;
          CeedInt index = (i * blk_size + elem) * layout[0] + j * layout[1] * blk_size + block * layout[2] * blk_size - blk_size * elem_size;
          if (y_array[index] != x_array[ind[k * elem_size + i]]) {
            // LCOV_EXCL_START
            printf("Error in restricted array y[%" CeedInt_FMT "][%" CeedInt_FMT "][%" CeedInt_FMT "] = %f\n", i, j, k, (double)y_array[index]);
            // LCOV_EXCL_STOP
          }
        }
      }
    }
    CeedVectorRestoreArrayRead(y, &y_array);
  }

  // Transpose
  CeedVectorSetValue(x, 0);
  CeedElemRestrictionApplyBlock(elem_restriction, 1, CEED_TRANSPOSE, y, x, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *x_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array);
    for (CeedInt i = blk_size; i < num_elem + 1; i++) {
      if (x_array[i] != (10 + i) * (i > blk_size && i < num_elem ? 2.0 : 1.0)) {
        // LCOV_EXCL_START
        printf("Error in restricted array x[%" CeedInt_FMT "] = %f\n", i, (double)x_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(x, &x_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
