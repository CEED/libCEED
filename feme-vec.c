#include <feme-impl.h>

int FemeVectorCreate(Feme feme, FemeInt length, FemeVector *vec) {
  int ierr;

  if (!feme->VecCreate) return FemeError(feme, 1, "Backend does not support VecCreate");
  ierr = FemeCalloc(1,vec);FemeChk(ierr);
  (*vec)->feme = feme;
  (*vec)->length = length;
  ierr = feme->VecCreate(feme, length, *vec);FemeChk(ierr);
  return 0;
}

int FemeVectorSetArray(FemeVector x, FemeMemType mtype, FemeCopyMode cmode, FemeScalar *array) {
  int ierr;

  if (!x || !x->SetArray) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->SetArray(x, mtype, cmode, array);FemeChk(ierr);
  return 0;
}

int FemeVectorGetArray(FemeVector x, FemeMemType mtype, FemeScalar **array) {
  int ierr;

  if (!x || !x->GetArray) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->GetArray(x, mtype, array);FemeChk(ierr);
  return 0;
}

int FemeVectorGetArrayRead(FemeVector x, FemeMemType mtype, const FemeScalar **array) {
  int ierr;

  if (!x || !x->GetArrayRead) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->GetArrayRead(x, mtype, array);FemeChk(ierr);
  return 0;
}

int FemeVectorRestoreArray(FemeVector x, FemeScalar **array) {
  int ierr;

  if (!x || !x->RestoreArray) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->RestoreArray(x, array);FemeChk(ierr);
  return 0;
}

int FemeVectorRestoreArrayRead(FemeVector x, const FemeScalar **array) {
  int ierr;

  if (!x || !x->RestoreArrayRead) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->RestoreArrayRead(x, array);FemeChk(ierr);
  return 0;
}

int FemeVectorDestroy(FemeVector *x) {
  int ierr;

  if (!*x) return 0;
  if ((*x)->Destroy) {
    ierr = (*x)->Destroy(*x);FemeChk(ierr);
  }
  ierr = FemeFree(x);FemeChk(ierr);
  return 0;
}
