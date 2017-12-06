#include <ceed-impl.h>

int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelem, CeedInt elemsize, CeedInt ndof, CeedMemType mtype, CeedCopyMode cmode, const CeedInt *indices, CeedElemRestriction *r) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) return CeedError(ceed, 1, "Backend does not support ElemRestrictionCreate");
  ierr = CeedCalloc(1,r);CeedChk(ierr);
  (*r)->ceed = ceed;
  (*r)->nelem = nelem;
  (*r)->elemsize = elemsize;
  (*r)->ndof = ndof;
  ierr = ceed->ElemRestrictionCreate(*r, mtype, cmode, indices);CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionApply(CeedElemRestriction r, CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt m,n;
  int ierr;

  if (tmode == CEED_NOTRANSPOSE) {
    m = r->nelem * r->elemsize;
    n = r->ndof;
  } else {
    m = r->ndof;
    n = r->nelem * r->elemsize;
  }
  if (n != u->length) return CeedError(r->ceed, 2, "Input vector size %d not compatible with element restriction (%d,%d)", u->length, r->nelem*r->elemsize, r->ndof);
  if (m != v->length) return CeedError(r->ceed, 2, "Output vector size %d not compatible with element restriction (%d,%d)", v->length, r->nelem*r->elemsize, r->ndof);
  ierr = r->Apply(r, tmode, u, v, request);CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionDestroy(CeedElemRestriction *r) {
  int ierr;

  if (!*r) return 0;
  if ((*r)->Destroy) {
    ierr = (*r)->Destroy(*r);CeedChk(ierr);
  }
  ierr = CeedFree(r);CeedChk(ierr);
  return 0;
}
