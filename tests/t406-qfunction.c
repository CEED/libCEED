/// @file
/// Test QFunction helper macro
/// \test Test QFunction helper macro
#include <ceed.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "t406-qfunction-helper.h"
#include "t406-qfunction.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector in[16], out[16];
  CeedVector Q_data, W, U, V;
  CeedQFunction qf_setup, qf_mass;
  CeedInt Q = 8;
  const CeedScalar *vv;
  CeedScalar w[Q], u[Q], v[Q];
  char *qf_source_paths;

  CeedInit(argv[1], &ceed);

  // Semicolon separated list of source files
  CeedConcatenateSourceFilePaths(ceed, &qf_source_paths, setup_loc, 1,
                                 multiply_helper_loc);

  CeedQFunctionCreateInterior(ceed, 1, setup, qf_source_paths, &qf_setup);
  free(qf_source_paths);
  CeedQFunctionAddInput(qf_setup, "w", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", 1, CEED_EVAL_NONE);

  // Semicolon separated list of source files
  CeedConcatenateSourceFilePaths(ceed, &qf_source_paths, mass_loc, 1,
                                 multiply_helper_loc);

  CeedQFunctionCreateInterior(ceed, 1, mass, qf_source_paths, &qf_mass);
  free(qf_source_paths);
  CeedQFunctionAddInput(qf_mass, "qdata", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = 2.*i/(Q-1) - 1;
    w[i] = 1 - x*x;
    u[i] = 2 + 3*x + 5*x*x;
    v[i] = w[i] * u[i];
  }

  CeedVectorCreate(ceed, Q, &W);
  CeedVectorSetArray(W, CEED_MEM_HOST, CEED_USE_POINTER, w);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);
  CeedVectorCreate(ceed, Q, &V);
  CeedVectorSetValue(V, 0);
  CeedVectorCreate(ceed, Q, &Q_data);
  CeedVectorSetValue(Q_data, 0);

  {
    in[0] = W;
    out[0] = Q_data;
    CeedQFunctionApply(qf_setup, Q, in, out);
  }
  {
    in[0] = W;
    in[1] = U;
    out[0] = V;
    CeedQFunctionApply(qf_mass, Q, in, out);
  }

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &vv);
  for (CeedInt i=0; i<Q; i++)
    if (fabs(5*v[i] - vv[i]) > 1E3*CEED_EPSILON)
      // LCOV_EXCL_START
      printf("[%d] v %f != vv %f\n",i, 5*v[i], vv[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(V, &vv);

  CeedVectorDestroy(&W);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&Q_data);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
