/// @file
/// Test creation, use, and destruction of an element restriction
/// \test Test creation, use, and destruction of an element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 3;
  CeedInt             ind[2 * num_elem];
  CeedScalar          a[num_elem + 1];
  const CeedScalar   *yy;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem + 1, &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, num_elem * 2, &y);
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  for (CeedInt i = 0; i < num_elem * 2; i++) {
    if (10 + (i + 1) / 2 != yy[i]) printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)yy[i]);
  }
  CeedVectorRestoreArrayRead(y, &yy);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
