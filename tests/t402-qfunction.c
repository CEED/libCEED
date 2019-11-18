/// @file
/// Test viewing of qfunction
/// \test Test viewing of qfunction
#include <ceed.h>

#include "t400-qfunction.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedQFunction qf_setup, qf_mass;


  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "w", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_INTERP);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  CeedQFunctionView(qf_setup, stdout);
  CeedQFunctionView(qf_mass, stdout);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
