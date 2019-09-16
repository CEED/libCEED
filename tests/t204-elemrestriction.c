/// @file
/// Test creation, use, and destruction of a multicomponent element restriction
/// \test Test creation, use, and destruction of a multicomponent element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 3;
  CeedInt ind[2*ne];
  CeedScalar a[2*(ne+1)];
  const CeedScalar *yy;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  // Setup
  CeedVectorCreate(ceed, 2*(ne+1), &x);
  for (CeedInt i=0; i<ne+1; i++) {
    a[i] = 10 + i;
    a[i+ne+1] = 20 + i;
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<ne; i++) {
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed, ne, 2, ne+1, 2, CEED_MEM_HOST,
                            CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, 2*(ne*2), &y);
  CeedVectorSetValue(y, 0); // Allocates array

  // Restrict
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, x, y,
                           CEED_REQUEST_IMMEDIATE);

  // Check
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  for (CeedInt i=0; i<ne; i++)
    for (CeedInt n=0; n<2; n++) {
      if (yy[i*4+n] != 10+(2*i+n+1)/2)
        // LCOV_EXCL_START
        printf("Error in restricted array y[%d] = %f != %f\n",
               i*4+n, (double)yy[i*4+n], 10.+(2*i+n+1)/2);
      // LCOV_EXCL_STOP
      if (yy[i*4+n+2] != 20+(2*i+n+1)/2)
        // LCOV_EXCL_START
        printf("Error in restricted array y[%d] = %f != %f\n",
               i*4+n+2, (double)yy[i*4+n+2], 20.+(2*i+n+1)/2);
      // LCOV_EXCL_STOP
    }

  CeedVectorRestoreArrayRead(y, &yy);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
