/// @file
/// Test CeedVectorSetValue
/// \test Test CeedVectorSetValue
#include <ceed.h>

static int CheckValues(Ceed ceed, CeedVector x, CeedScalar value) {
  const CeedScalar *b;
  CeedSize          n;
  CeedVectorGetLength(x, &n);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (b[i] != value) printf("Error reading array b[%" CeedInt_FMT "] = %f", i, (CeedScalar)b[i]);
  }
  CeedVectorRestoreArrayRead(x, &b);
  return 0;
}

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

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (b[i] != 10 + i) printf("Error reading array b[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)b[i]);
  }
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorSetValue(x, 3.0);
  CheckValues(ceed, x, 3.0);
  CeedVectorDestroy(&x);

  CeedVectorCreate(ceed, n, &x);
  // Set value before setting or getting the array
  CeedVectorSetValue(x, 5.0);
  CheckValues(ceed, x, 5.0);
  CeedVectorDestroy(&x);

  CeedDestroy(&ceed);
  return 0;
}
