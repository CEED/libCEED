#include <feme-impl.h>

int FemeElemRestrictionCreate(Feme feme, FemeInt nelem, FemeInt elemsize, FemeInt ndof, FemeMemType mtype, FemeCopyMode cmode, const FemeInt *indices, FemeElemRestriction *r) {
  int ierr;

  if (!feme->ElemRestrictionCreate) return FemeError(feme, 1, "Backend does not support ElemRestrictionCreate");
  ierr = FemeCalloc(1,r);FemeChk(ierr);
  (*r)->feme = feme;
  (*r)->nelem = nelem;
  (*r)->elemsize = elemsize;
  (*r)->ndof = ndof;
  ierr = feme->ElemRestrictionCreate(*r, mtype, cmode, indices);FemeChk(ierr);
  return 0;
}

int FemeElemRestrictionApply(FemeElemRestriction r, FemeTransposeMode tmode, FemeVec u, FemeVec v, FemeRequest *request) {
  FemeInt m,n;
  int ierr;

  if (tmode == FEME_NOTRANSPOSE) {
    m = r->nelem * r->elemsize;
    n = r->ndof;
  } else {
    m = r->ndof;
    n = r->nelem * r->elemsize;
  }
  if (n != u->n) return FemeError(r->feme, 2, "Input vector size %d not compatible with element restriction (%d,%d)", u->n, r->nelem*r->elemsize, r->ndof);
  if (m != v->n) return FemeError(r->feme, 2, "Output vector size %d not compatible with element restriction (%d,%d)", v->n, r->nelem*r->elemsize, r->ndof);
  ierr = r->Apply(r, tmode, u, v, request);FemeChk(ierr);
  return 0;
}

int FemeElemRestrictionDestroy(FemeElemRestriction *r) {
  int ierr;

  if (!*r) return 0;
  if ((*r)->Destroy) {
    ierr = (*r)->Destroy(*r);FemeChk(ierr);
  }
  ierr = FemeFree(r);FemeChk(ierr);
  return 0;
}
