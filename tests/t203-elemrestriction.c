/// @file
/// Test creation, use, and destruction of a blocked element restriction with multiple components in the lvector
/// \test Test creation, use, and destruction of a blocked element restriction with multiple components in the lvector
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem  = 8;
  CeedInt             elem_size = 2;
  CeedInt             num_blk   = 2;
  CeedInt             blk_size  = 5;
  CeedInt             num_comp  = 3;
  CeedInt             ind[elem_size * num_elem];
  CeedScalar          x_array[num_comp * (num_elem + 1)];
  CeedInt             layout[3];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_comp * (num_elem + 1), &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) {
    x_array[i + 0 * (num_elem + 1)] = 10 + i;
    x_array[i + 1 * (num_elem + 1)] = 20 + i;
    x_array[i + 2 * (num_elem + 1)] = 30 + i;
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, num_comp * num_blk * blk_size * elem_size, &y);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, blk_size, num_comp, num_elem + 1, num_comp * (num_elem + 1), CEED_MEM_HOST,
                                   CEED_USE_POINTER, ind, &elem_restriction);

  // NoTranspose
  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    CeedElemRestrictionGetELayout(elem_restriction, &layout);
    for (CeedInt i = 0; i < elem_size; i++) {     // Node
      for (CeedInt j = 0; j < num_comp; j++) {    // Component
        for (CeedInt k = 0; k < num_elem; k++) {  // Element
          CeedInt block = k / blk_size;
          CeedInt elem  = k % blk_size;
          CeedInt index = (i * blk_size + elem) * layout[0] + j * layout[1] * blk_size + block * layout[2] * blk_size;
          if (y_array[index] != x_array[ind[k * elem_size + i] + j * (num_elem + 1)]) {
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
  CeedElemRestrictionApply(elem_restriction, CEED_TRANSPOSE, y, x, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *x_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array);
    for (CeedInt i = 0; i < num_elem + 1; i++) {
      for (CeedInt j = 0; j < num_comp; j++) {
        if (x_array[i + j * (num_elem + 1)] != ((j + 1) * 10 + i) * (i > 0 && i < num_elem ? 2.0 : 1.0)) {
          // LCOV_EXCL_START
          printf("Error in restricted array x[%" CeedInt_FMT "][%" CeedInt_FMT "] = %f\n", j, i, (double)x_array[i + j * (num_elem + 1)]);
          // LCOV_EXCL_STOP
        }
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
