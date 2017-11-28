#include <feme-impl.h>

int FemeQFunctionCreateInterior(Feme feme, FemeInt vlength, FemeInt nfields, size_t qdatasize, FemeEvalMode inmode, FemeEvalMode outmode,
                                int (*f)(void*, void*, FemeInt, const FemeScalar *const*, FemeScalar *const*),
                                const char *focca, FemeQFunction *qf) {
  int ierr;

  if (!feme->QFunctionCreate) return FemeError(feme, 1, "Backend does not support QFunctionCreate");
  ierr = FemeCalloc(1,qf);FemeChk(ierr);
  (*qf)->feme = feme;
  (*qf)->vlength = vlength;
  (*qf)->nfields = nfields;
  (*qf)->qdatasize = qdatasize;
  (*qf)->inmode = inmode;
  (*qf)->outmode = outmode;
  (*qf)->function = f;
  (*qf)->focca = focca;
  ierr = feme->QFunctionCreate(*qf);FemeChk(ierr);
  return 0;
}

int FemeQFunctionSetContext(FemeQFunction qf, void *ctx, size_t ctxsize) {
  qf->ctx = ctx;
  qf->ctxsize = ctxsize;
  return 0;
}

int FemeQFunctionDestroy(FemeQFunction *qf) {
  int ierr;

  if (!*qf) return 0;
  if ((*qf)->Destroy) {
    ierr = (*qf)->Destroy(*qf);FemeChk(ierr);
  }
  ierr = FemeFree(qf);FemeChk(ierr);
  return 0;
}
