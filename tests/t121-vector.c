/// @file
/// Test summing of a pair of vectors
/// \test Test summing of a pair of vectors
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x, y;
  CeedInt           n;
  CeedScalar        a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorCreate(ceed, n, &y);
  for (CeedInt i = 0; i < n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);
  CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, a);

  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(y, type);
  }
  CeedVectorAXPY(y, -0.5, x);

  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (fabs(b[i] - (10.0 + i) / 2) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in alpha x + y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, b[i], (10.0 + i) / 2);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(y, &b);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
