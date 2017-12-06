#include <ceed-impl.h>

int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength, CeedInt nfields, size_t qdatasize, CeedEvalMode inmode, CeedEvalMode outmode,
                                int (*f)(void*, void*, CeedInt, const CeedScalar *const*, CeedScalar *const*),
                                const char *focca, CeedQFunction *qf) {
  int ierr;

  if (!ceed->QFunctionCreate) return CeedError(ceed, 1, "Backend does not support QFunctionCreate");
  ierr = CeedCalloc(1,qf);CeedChk(ierr);
  (*qf)->ceed = ceed;
  (*qf)->vlength = vlength;
  (*qf)->nfields = nfields;
  (*qf)->qdatasize = qdatasize;
  (*qf)->inmode = inmode;
  (*qf)->outmode = outmode;
  (*qf)->function = f;
  (*qf)->focca = focca;
  ierr = ceed->QFunctionCreate(*qf);CeedChk(ierr);
  return 0;
}

int CeedQFunctionSetContext(CeedQFunction qf, void *ctx, size_t ctxsize) {
  qf->ctx = ctx;
  qf->ctxsize = ctxsize;
  return 0;
}

int CeedQFunctionDestroy(CeedQFunction *qf) {
  int ierr;

  if (!*qf) return 0;
  if ((*qf)->Destroy) {
    ierr = (*qf)->Destroy(*qf);CeedChk(ierr);
  }
  ierr = CeedFree(qf);CeedChk(ierr);
  return 0;
}
