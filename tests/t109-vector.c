/// @file
/// Test CeedVectorSetArray to remove array access
/// \test Test CeedVectorSetArray to remove array access
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x;
  const CeedInt     n = 10;
  CeedScalar        a[n];
  CeedScalar       *b, *c;
  const CeedScalar *d;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i = 0; i < n; i++) a[i] = 0;
  a[3] = -3.14;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  // Taking array should return a
  CeedVectorTakeArray(x, CEED_MEM_HOST, &c);
  if (fabs(c[3] + 3.14) > 10. * CEED_EPSILON) printf("Error taking array c[3] = %f\n", (CeedScalar)c[3]);

  // Getting array should not modify a
  CeedVectorGetArrayWrite(x, CEED_MEM_HOST, &b);
  b[5] = -3.14;
  CeedVectorRestoreArray(x, &b);

  if (fabs(a[5] + 3.14) < 10. * CEED_EPSILON) printf("Error protecting array a[3] = %f\n", (CeedScalar)a[3]);

  // Note: We do not need to free c because c == a was stack allocated.
  CeedVectorDestroy(&x);

  // Test with a size zero vector
  CeedVectorCreate(ceed, 0, &x);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, NULL);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &d);
  if (b) printf("CeedVectorGetArrayRead returned non-NULL for zero-sized Vector\n");
  CeedVectorRestoreArrayRead(x, &d);
  CeedVectorTakeArray(x, CEED_MEM_HOST, &c);
  if (c) printf("CeedVectorTakeArray returned non-NULL for zero-sized Vector\n");
  CeedVectorDestroy(&x);

  CeedDestroy(&ceed);
  return 0;
}
