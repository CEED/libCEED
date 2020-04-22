/// @file
/// Test creation, use, and destruction of a blocked element restriction
/// \test Test creation, use, and destruction of a blocked element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 8;
  CeedInt blksize = 5;
  CeedInt elemsize = 2;
  CeedInt ind[elemsize*ne];
  CeedScalar a[ne+1];
  CeedElemRestriction r;
  CeedScalar *y_array;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, ne+1, &x);
  for (CeedInt i=0; i<ne+1; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<ne; i++) {
    for (CeedInt k=0; k<elemsize; k++) {
      ind[elemsize*i+k] = i+k;
    }
  }

  CeedElemRestrictionCreateBlocked(ceed, ne, elemsize, blksize, 1, 1, ne+1,
                                   CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);

  CeedVectorCreate(ceed, blksize*elemsize, &y);
  CeedVectorSetValue(y, 0); // Allocates array

  // NoTranspose
  CeedElemRestrictionApplyBlock(r, 1, CEED_NOTRANSPOSE, x, y,
                                CEED_REQUEST_IMMEDIATE);

  // Zero padded entries
  CeedVectorGetArray(y, CEED_MEM_HOST, &y_array);
  for (CeedInt i = (elemsize*ne - blksize*elemsize); i < blksize*elemsize; ++i) {
    y_array[i] = 0;
  }
  CeedVectorRestoreArray(y, &y_array);
  CeedVectorView(y, "%12.8f", stdout);

  // Transpose
  CeedVectorGetArray(x, CEED_MEM_HOST, (CeedScalar **)&a);
  for (CeedInt i=0; i<ne+1; i++)
    a[i] = 0;
  CeedVectorRestoreArray(x, (CeedScalar **)&a);
  CeedElemRestrictionApplyBlock(r, 1, CEED_TRANSPOSE, y, x,
                                CEED_REQUEST_IMMEDIATE);
  CeedVectorView(x, "%12.8f", stdout);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
