/// @file
/// Test creation, use, and destruction of an identity element restriction
/// \test Test creation, use, and destruction of an identity element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 3;
  CeedScalar a[ne*2];
  const CeedScalar *yy;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);
  CeedVectorCreate(ceed, ne*2, &x);
  for (CeedInt i=0; i<ne*2; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedElemRestrictionCreateIdentity(ceed, ne, 2, ne*2, 1, &r);
  CeedVectorCreate(ceed, ne*2, &y);
  CeedVectorSetValue(y, 0); // Allocates array
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, x, y,
                           CEED_REQUEST_IMMEDIATE);
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  for (CeedInt i=0; i<ne*2; i++)
    if (yy[i] != 10+i)
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
