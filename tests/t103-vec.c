/// @file
/// Test setting one vector from array of another vector
/// \test Test setting one vector from array of another vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector X, Y;
  CeedInt n;
  CeedScalar a[10];
  CeedScalar *x;
  const CeedScalar *y;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &X);
  CeedVectorCreate(ceed, n, &Y);

  for (CeedInt i=0; i<n; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedVectorGetArray(X, CEED_MEM_HOST, &x);
  CeedVectorSetArray(Y, CEED_MEM_HOST, CEED_COPY_VALUES, x);
  CeedVectorRestoreArray(X, &x);

  CeedVectorGetArrayRead(Y, CEED_MEM_HOST, &y);
  for (CeedInt i=0; i<n; i++)
    if (y[i] != 10+i)
      // LCOV_EXCL_START
      printf("Error reading array y[%d] = %f",i,(double)y[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(Y, &y);

  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Y);
  CeedDestroy(&ceed);
  return 0;
}
