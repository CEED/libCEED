#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar a[10];
  const CeedScalar *b;

  CeedInit("/cpu/self", &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++) a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++) {
    if (10+i != b[i])
      return CeedError(ceed, (int)i, "Error reading array b[%d] = %f",i,
                       (double)b[i]);
  }
  CeedVectorRestoreArrayRead(x, &b);
  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
