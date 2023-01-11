/// @file
/// Test CeedVectorGetArray to modify array
/// \test Test CeedVectorGetArray to modify array
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    x;
  const CeedInt n = 10;
  CeedScalar    a[n];
  CeedScalar   *b;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i = 0; i < n; i++) a[i] = 0;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedVectorGetArray(x, CEED_MEM_HOST, &b);
  b[3] = -3.14;
  CeedVectorRestoreArray(x, &b);

  if (a[3] != (CeedScalar)(-3.14)) printf("Error writing array a[3] = %f\n", (CeedScalar)a[3]);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
