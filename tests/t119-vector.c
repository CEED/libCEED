/// @file
/// Test taking the reciprocal of a vector
/// \test Test taking the reciprocal of a vector
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x;
  CeedInt           n;
  CeedScalar        a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i = 0; i < n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }
  CeedVectorReciprocal(x);

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (fabs(b[i] - 1. / (10 + i)) > 10. * CEED_EPSILON) printf("Error reading array b[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)b[i]);
  }
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
