#include <feme-impl.h>


int FemeOperatorCreate(Feme feme, FemeElemRestriction r, FemeBasis b, FemeQFunction qf, FemeQFunction dqf, FemeQFunction dqfT, FemeOperator *op) {
  int ierr;

  if (!feme->OperatorCreate) return FemeError(feme, 1, "Backend does not support OperatorCreate");
  ierr = FemeCalloc(1,op);FemeChk(ierr);
  (*op)->feme = feme;
  (*op)->Erestrict = r;
  (*op)->basis = b;
  (*op)->qf = qf;
  (*op)->dqf = dqf;
  (*op)->dqfT = dqfT;
  ierr = feme->OperatorCreate(*op);FemeChk(ierr);
  return 0;
}

int FemeOperatorApply(FemeOperator op, FemeVector qdata, FemeVector ustate, FemeVector residual, FemeRequest *request) {
  int ierr;

  ierr = op->Apply(op, qdata, ustate, residual, request);FemeChk(ierr);
  return 0;
}

int FemeOperatorDestroy(FemeOperator *op) {
  int ierr;

  if (!*op) return 0;
  if ((*op)->Destroy) {
    ierr = (*op)->Destroy(*op);FemeChk(ierr);
  }
  ierr = FemeFree(op);FemeChk(ierr);
  return 0;
}
