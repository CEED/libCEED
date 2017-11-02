#include <feme.h>

int main(int argc, char **argv) {
  Feme feme;
  FemeVec x;
  FemeInt n;
  FemeScalar a[10];
  const FemeScalar *b;

  FemeInit("/cpu/self", &feme);
  n = 10;
  FemeVecCreate(feme, n, &x);
  for (FemeInt i=0; i<n; i++) a[i] = 10 + i;
  FemeVecSetArray(x, FEME_MEM_HOST, FEME_USE_POINTER, a);
  FemeVecGetArrayRead(x, FEME_MEM_HOST, &b);
  for (FemeInt i=0; i<n; i++) {
    if (10+i != b[i]) FemeError(feme, (int)i, "Error reading array b[%d] = %f",i,(double)b[i]);
  }
  FemeVecRestoreArrayRead(x, &b);
  FemeVecDestroy(&x);
  FemeDestroy(&feme);
  return 0;
}
