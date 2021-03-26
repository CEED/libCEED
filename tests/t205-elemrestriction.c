/// @file
/// Test creation, use, and destruction of an interlaced multicomponent element restriction
/// \test Test creation, use, and destruction of an interlaced multicomponent element restriction
#include <ceed.h>
#include <ceed/backend.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 3;
  CeedInt ind[2*ne];
  CeedInt layout[3];
  CeedScalar a[2*(ne+1)];
  const CeedScalar *yy;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  // Setup
  CeedVectorCreate(ceed, 2*(ne+1), &x);
  for (CeedInt i=0; i<ne+1; i++) {
    a[2*i] = 10 + i;
    a[2*i+1] = 20 + i;
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<ne; i++) {
    ind[2*i+0] = 2*i;
    ind[2*i+1] = 2*(i+1);
  }
  CeedElemRestrictionCreate(ceed, ne, 2, 2, 1, 2*(ne+1), CEED_MEM_HOST,
                            CEED_USE_POINTER, ind, &r);
  CeedVectorCreate(ceed, 2*(ne*2), &y);
  CeedVectorSetValue(y, 0); // Allocates array

  // Restrict
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);

  // Check
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  CeedElemRestrictionGetELayout(r, &layout);
  for (CeedInt i=0; i<2; i++)       // Node
    for (CeedInt j=0; j<2; j++)     // Component
      for (CeedInt k=0; k<ne; k++)  // Element
        if (yy[i*layout[0]+j*layout[1]+k*layout[2]] != a[ind[i+k*2]+j])
          // LCOV_EXCL_START
          printf("Error in restricted array y[%d][%d][%d] = %f != %f\n",
                 i, j, k, (double)yy[i*layout[0]+j*layout[1]+k*layout[2]],
                 a[ind[i+k*2]+j*(ne+1)]);
  // LCOV_EXCL_STOP

  CeedVectorRestoreArrayRead(y, &yy);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
