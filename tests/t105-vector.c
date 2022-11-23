/// @file
/// Test creation, setting, reading, restoring, and destroying of a vector using CEED_MEM_DEVICE
/// \test Test creation, setting, reading, restoring, and destroying of a vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x;
  CeedVector        y;
  CeedInt           n;
  CeedScalar        a[10];
  const CeedScalar *b, *c;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorCreate(ceed, n, &y);
  for (CeedInt i = 0; i < n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedVectorGetArrayRead(x, CEED_MEM_DEVICE, &b);
  CeedVectorSetArray(y, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)b);
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &c);
  for (CeedInt i = 0; i < n; i++) {
    if (c[i] != 10 + i) printf("Error reading array c[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)c[i]);
  }
  CeedVectorRestoreArrayRead(y, &c);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
