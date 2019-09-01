/// @file
/// Test creation, use, and destruction of an element restriction
/// \test Test creation, use, and destruction of an element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 3;
  CeedInt ind[2*ne];
  CeedScalar a[ne+1];
  const CeedScalar *yy;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);
  CeedVectorCreate(ceed, ne+1, &x);
  for (CeedInt i=0; i<ne+1; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<ne; i++) {
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed, ne, 2, ne+1, 1, CEED_MEM_HOST, CEED_USE_POINTER,
                            ind, &r);
  CeedVectorCreate(ceed, ne*2, &y);
  CeedVectorSetValue(y, 0); // Allocates array
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, x, y,
                           CEED_REQUEST_IMMEDIATE);
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  for (CeedInt i=0; i<ne*2; i++)
    if (10+(i+1)/2 != yy[i])
      // LCOV_EXCL_START
      printf("Error in restricted array y[%d] = %f",
             i, (double)yy[i]);
      // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(y, &yy);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
