/// @file
/// Test creation, use, and destruction of a blocked element restriction
/// \test Test creation, use, and destruction of a blocked element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt num_elem = 8;
  CeedInt blk_size = 5;
  CeedInt ind[2*num_elem];
  CeedScalar a[num_elem+1];
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem+1, &x);
  for (CeedInt i=0; i<num_elem+1; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<num_elem; i++) {
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  CeedElemRestrictionCreateBlocked(ceed, num_elem, 2, blk_size, 1, 1, num_elem+1,
                                   CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, 2*blk_size*2, &y);
  CeedVectorSetValue(y, 0); // Allocates array

  // NoTranspose
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  CeedVectorView(y, "%12.8f", stdout);

  // Transpose
  CeedVectorGetArray(x, CEED_MEM_HOST, (CeedScalar **)&a);
  for (CeedInt i=0; i<num_elem+1; i++)
    a[i] = 0;
  CeedVectorRestoreArray(x, (CeedScalar **)&a);
  CeedElemRestrictionApply(r, CEED_TRANSPOSE, y, x, CEED_REQUEST_IMMEDIATE);
  CeedVectorView(x, "%12.8f", stdout);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
