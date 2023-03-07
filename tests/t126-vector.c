/// @file
/// Test creation, copying, and destroying of a vector
/// \test Test creation, copying, and destroying of a vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, x_copy;
  CeedInt    n;
  CeedScalar        a[10], a2[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorCreate(ceed, n, &x_copy);

  for (CeedInt i = 0; i < n; i++) {
    a[i] = 10 + i;
    a2[i] = i;
  }
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);
  CeedVectorSetArray(x_copy, CEED_MEM_HOST, CEED_COPY_VALUES, a2);

  CeedVectorCopy(x, &x_copy);

  CeedSize len;
  CeedVectorGetLength(x_copy, &len);
  if (len != n) printf("Error copying CeedVector: %td\n", len);

  // Check that new array from x_copy is the same as the original input array a
  CeedVectorGetArrayRead(x_copy, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (a[i] != b[i]) printf("Error in copying values of CeedVector: %f, %f\n", a[i], b[i]);
  }
  CeedVectorRestoreArrayRead(x_copy, &b);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&x_copy);
  CeedDestroy(&ceed);
  return 0;
}
