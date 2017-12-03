#include <feme.h>

int main(int argc, char **argv) {
  Feme feme;
  FemeVector x, y;
  FemeInt ne = 3;
  FemeInt ind[2*ne];
  FemeScalar a[ne+1];
  const FemeScalar *yy;
  FemeElemRestriction r;

  FemeInit("/cpu/self", &feme);
  FemeVectorCreate(feme, ne+1, &x);
  for (FemeInt i=0; i<ne+1; i++) a[i] = 10 + i;
  FemeVectorSetArray(x, FEME_MEM_HOST, FEME_USE_POINTER, a);

  for (FemeInt i=0; i<ne; i++) {
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  FemeElemRestrictionCreate(feme, ne, 2, ne+1, FEME_MEM_HOST, FEME_USE_POINTER, ind, &r);
  FemeVectorCreate(feme, ne*2, &y);
  FemeVectorSetArray(y, FEME_MEM_HOST, FEME_COPY_VALUES, NULL); // Allocates array
  FemeElemRestrictionApply(r, FEME_NOTRANSPOSE, x, y, FEME_REQUEST_IMMEDIATE);
  FemeVectorGetArrayRead(y, FEME_MEM_HOST, &yy);
  for (FemeInt i=0; i<ne*2; i++) {
    if (10+(i+1)/2 != yy[i]) FemeError(feme, (int)i, "Error in restricted array y[%d] = %f",i,(double)yy[i]);
  }
  FemeVectorRestoreArrayRead(y, &yy);
  FemeVectorDestroy(&x);
  FemeVectorDestroy(&y);
  FemeElemRestrictionDestroy(&r);
  FemeDestroy(&feme);
  return 0;
}
