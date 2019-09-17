/// @file
/// Test creation, use, and destruction of a blocked element restriction with multiple components in the lvector
/// \test Test creation, use, and destruction of a blocked element restriction with multiple components in the lvector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 8;
  CeedInt blksize = 5;
  CeedInt ncomp = 3;
  CeedInt ind[2*ne];
  CeedScalar a[ncomp*(ne+1)];
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, (ne+1)*ncomp, &x);
  for (CeedInt i=0; i<(ne+1); i++) {
    a[i+0*(ne+1)] = 10 + i;
    a[i+1*(ne+1)] = 20 + i;
    a[i+2*(ne+1)] = 30 + i;
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  CeedVectorView(x, "%12.8f", stdout);
  for (CeedInt i=0; i<ne; i++) {
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  CeedElemRestrictionCreateBlocked(ceed, ne, 2, blksize, ne+1, ncomp,
                                   CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, 2*blksize*2*ncomp, &y);
  CeedVectorSetValue(y, 0); // Allocates array

  // NoTranspose
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, x, y,
                           CEED_REQUEST_IMMEDIATE);
  CeedVectorView(y, "%12.8f", stdout);

  // Transpose
  CeedVectorGetArray(x, CEED_MEM_HOST, (CeedScalar **)&a);
  for (CeedInt i=0; i<(ne+1)*ncomp; i++)
    a[i] = 0;
  CeedVectorRestoreArray(x, (CeedScalar **)&a);
  CeedElemRestrictionApply(r, CEED_TRANSPOSE, CEED_NOTRANSPOSE, y, x,
                           CEED_REQUEST_IMMEDIATE);
  CeedVectorView(x, "%12.8f", stdout);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
