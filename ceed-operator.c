#include <ceed-impl.h>


int CeedOperatorCreate(Ceed ceed, CeedElemRestriction r, CeedBasis b, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT, CeedOperator *op) {
  int ierr;

  if (!ceed->OperatorCreate) return CeedError(ceed, 1, "Backend does not support OperatorCreate");
  ierr = CeedCalloc(1,op);CeedChk(ierr);
  (*op)->ceed = ceed;
  (*op)->Erestrict = r;
  (*op)->basis = b;
  (*op)->qf = qf;
  (*op)->dqf = dqf;
  (*op)->dqfT = dqfT;
  ierr = ceed->OperatorCreate(*op);CeedChk(ierr);
  return 0;
}

int CeedOperatorApply(CeedOperator op, CeedVector qdata, CeedVector ustate, CeedVector residual, CeedRequest *request) {
  int ierr;

  ierr = op->Apply(op, qdata, ustate, residual, request);CeedChk(ierr);
  return 0;
}

int CeedOperatorDestroy(CeedOperator *op) {
  int ierr;

  if (!*op) return 0;
  if ((*op)->Destroy) {
    ierr = (*op)->Destroy(*op);CeedChk(ierr);
  }
  ierr = CeedFree(op);CeedChk(ierr);
  return 0;
}
