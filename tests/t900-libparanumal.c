/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedQFunction qf;
  CeedOperator op;
  CeedElemRestriction ErestrictId, Erestrictq;
  CeedBasis bq;
  CeedVector ggeo, S, MM, q, Aq;
  CeedInt nelem = 15, P = 5, Q = 8;

  CeedInit(argv[1], &ceed);
  CeedQFunctionCreateInteriorFromGallery(ceed, 1, "elliptic", &qf);
  CeedOperatorCreate(ceed, qf, NULL, NULL, &op);

  CeedElemRestrictionCreateIdentity(ceed, nelem, Q, Q*nelem, 1, &ErestrictId);
  CeedElemRestrictionCreateIdentity(ceed, nelem, P, P*nelem, 1, &Erestrictq);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &bq);

  CeedOperatorSetField(op, "ggeo", ErestrictId, CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED, ggeo);
  CeedOperatorSetField(op, "S"   , ErestrictId, CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED, S);
  CeedOperatorSetField(op, "MM"  , ErestrictId, CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED, MM);
  CeedOperatorSetField(op, "q"   , Erestrictq , CEED_NOTRANSPOSE, bq                   , CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op, "Aq"  , Erestrictq , CEED_NOTRANSPOSE, bq                   , CEED_VECTOR_ACTIVE);

  CeedQFunctionDestroy(&qf);
  CeedOperatorDestroy(&op);
  CeedDestroy(&ceed);
  return 0;
}
