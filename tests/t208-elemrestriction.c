/// @file
/// Test creation, use, and destruction of a blocked element restriction
/// \test Test creation, use, and destruction of a blocked element restriction
#include <ceed.h>
#include <ceed/backend.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem  = 8;
  CeedInt             elem_size = 2;
  CeedInt             blk_size  = 5;
  CeedInt             ind[elem_size * num_elem];
  CeedScalar          a[num_elem + 1];
  const CeedScalar   *xx, *yy;
  CeedInt             layout[3];
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem + 1, &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, blk_size, 1, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, blk_size * elem_size, &y);

  // NoTranspose
  CeedElemRestrictionApplyBlock(r, 1, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  CeedElemRestrictionGetELayout(r, &layout);
  for (CeedInt i = 0; i < elem_size; i++) {            // Node
    for (CeedInt j = 0; j < 1; j++) {                  // Component
      for (CeedInt k = blk_size; k < num_elem; k++) {  // Element
        CeedInt block = k / blk_size;
        CeedInt elem  = k % blk_size;
        CeedInt index = (i * blk_size + elem) * layout[0] + j * layout[1] * blk_size + block * layout[2] * blk_size - blk_size * elem_size;
        if (yy[index] != a[ind[k * elem_size + i]]) {
          // LCOV_EXCL_START
          printf("Error in restricted array y[%" CeedInt_FMT "][%" CeedInt_FMT "][%" CeedInt_FMT "] = %f\n", i, j, k, (double)yy[index]);
          // LCOV_EXCL_STOP
        }
      }
    }
  }
  CeedVectorRestoreArrayRead(y, &yy);

  // Transpose
  CeedVectorSetValue(x, 0);
  CeedElemRestrictionApplyBlock(r, 1, CEED_TRANSPOSE, y, x, CEED_REQUEST_IMMEDIATE);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &xx);
  for (CeedInt i = blk_size; i < num_elem + 1; i++) {
    if (xx[i] != (10 + i) * (i > blk_size && i < num_elem ? 2.0 : 1.0)) {
      // LCOV_EXCL_START
      printf("Error in restricted array x[%" CeedInt_FMT "] = %f\n", i, (double)xx[i]);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(x, &xx);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
