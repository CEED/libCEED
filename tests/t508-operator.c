/// @file
/// Test creation, copying, and destruction for mass matrix operator
/// \test Test creation, copying, and destruction for mass matrix operator
#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "t500-operator.h"

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedQFunction qf, qf_2;
  CeedOperator  op, op_2;

  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf);
  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_2);
  CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op);
  CeedOperatorCreate(ceed, qf_2, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_2);

  CeedOperatorReferenceCopy(op, &op_2);  // This destroys the previous op_2
  if (op != op_2) printf("Error copying CeedOperator reference\n");

  CeedQFunctionDestroy(&qf);
  CeedQFunctionDestroy(&qf_2);
  CeedOperatorDestroy(&op);
  CeedOperatorDestroy(&op_2);
  CeedDestroy(&ceed);
  return 0;
}
