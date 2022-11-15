/// @file
/// Test creation, use, and destruction of an element restriction oriented
/// \test Test creation, use, and destruction of an element restriction oriented
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 6, P = 2, dim = 1;
  CeedInt             ind[P * num_elem];
  bool                orient[P * num_elem];
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
    // flip the dofs on element 1,3,...
    orient[2 * i + 0] = (i % (2)) * -1 < 0;
    orient[2 * i + 1] = (i % (2)) * -1 < 0;
  }
  CeedElemRestrictionCreateOriented(ceed, num_elem, P, dim, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, orient, &r);
  CeedVectorCreate(ceed, num_elem * 2, &y);
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < P; j++) {
      CeedInt k = j + P * i;
      if (10 + (k + 1) / 2 != yy[k] * CeedIntPow(-1, i % 2)) {
        // LCOV_EXCL_START
        printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)yy[k]);
        // LCOV_EXCL_STOP
      }
    }
  }
  CeedVectorRestoreArrayRead(y, &yy);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
