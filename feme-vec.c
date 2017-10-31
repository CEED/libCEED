#include <feme-impl.h>

int FemeVecCreate(Feme feme, FemeInt n, FemeVec *vec) {
  int ierr;

  if (!feme->VecCreate) return FemeError(feme, 1, "Backend does not support VecCreate");
  ierr = FemeCalloc(1,vec);FemeChk(ierr);
  (*vec)->feme = feme;
  (*vec)->n = n;
  ierr = feme->VecCreate(feme, n, *vec);FemeChk(ierr);
  return 0;
}

int FemeVecSetArray(FemeVec x, FemeMemType mtype, FemeCopyMode cmode, FemeScalar *array) {
  int ierr;

  if (!x || !x->SetArray) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->SetArray(x, mtype, cmode, array);FemeChk(ierr);
  return 0;
}

int FemeVecGetArray(FemeVec x, FemeMemType mtype, FemeScalar **array) {
  int ierr;

  if (!x || !x->GetArray) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->GetArray(x, mtype, array);FemeChk(ierr);
  return 0;
}

int FemeVecGetArrayRead(FemeVec x, FemeMemType mtype, const FemeScalar **array) {
  int ierr;

  if (!x || !x->GetArrayRead) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->GetArrayRead(x, mtype, array);FemeChk(ierr);
  return 0;
}

int FemeVecRestoreArray(FemeVec x, FemeScalar **array) {
  int ierr;

  if (!x || !x->RestoreArray) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->RestoreArray(x, array);FemeChk(ierr);
  return 0;
}

int FemeVecRestoreArrayRead(FemeVec x, const FemeScalar **array) {
  int ierr;

  if (!x || !x->RestoreArrayRead) return FemeError(x ? x->feme : NULL, 1, "Not supported");
  ierr = x->RestoreArrayRead(x, array);FemeChk(ierr);
  return 0;
}

int FemeVecDestroy(FemeVec *x) {
  int ierr;

  if (!*x) return 0;
  if ((*x)->Destroy) {
    ierr = (*x)->Destroy(*x);FemeChk(ierr);
  }
  ierr = FemeFree(x);FemeChk(ierr);
  return 0;
}
