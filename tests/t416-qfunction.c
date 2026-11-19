/// @file
/// Test QFunctions with identical JiT sources
/// \test Test QFunctions with identical JiT sources
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "t406-qfunction.h"

// Apply the setup and mass QFunctions of t406, created anew, and check the result
static void ApplyMass(Ceed ceed, CeedInt q) {
  CeedVector    in[16], out[16];
  CeedVector    q_data, w, u, v;
  CeedQFunction qf_setup, qf_mass;
  CeedScalar    v_true[q];

  CeedVectorCreate(ceed, q, &w);
  CeedVectorCreate(ceed, q, &u);
  {
    CeedScalar w_array[q], u_array[q];

    for (CeedInt i = 0; i < q; i++) {
      CeedScalar x = 2. * i / (q - 1) - 1;
      w_array[i]   = 1 - x * x;
      u_array[i]   = 2 + 3 * x + 5 * x * x;
      v_true[i]    = w_array[i] * u_array[i];
    }
    CeedVectorSetArray(w, CEED_MEM_HOST, CEED_COPY_VALUES, w_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }
  CeedVectorCreate(ceed, q, &v);
  CeedVectorSetValue(v, 0);
  CeedVectorCreate(ceed, q, &q_data);
  CeedVectorSetValue(q_data, 0);

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "w", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "q data", 1, CEED_EVAL_NONE);
  {
    in[0]  = w;
    out[0] = q_data;
    CeedQFunctionApply(qf_setup, q, in, out);
  }

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q data", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);
  {
    in[0]  = q_data;
    in[1]  = u;
    out[0] = v;
    CeedQFunctionApply(qf_mass, q, in, out);
  }

  // Verify results
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < q; i++) {
      if (fabs(5 * COMPILER_DEFINED_SCALE * v_true[i] * sqrt(2.) - v_array[i]) > 5E3 * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT "] v_true %f != v %f\n", i, 5 * COMPILER_DEFINED_SCALE * v_true[i] * sqrt(2.), v_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&w);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
}

// Initialize a Ceed for the QFunctions of t406, with an optional extra JiT define
static void InitCeed(const char *resource, const char *extra_define, Ceed *ceed) {
  char  file_path[2056] = __FILE__;
  char *last_slash      = strrchr(file_path, '/');

  CeedInit(resource, ceed);
  memcpy(&file_path[last_slash - file_path], "/test-include/", 15);
  CeedAddJitSourceRoot(*ceed, file_path);
  CeedAddJitDefine(*ceed, "COMPILER_DEFINED_SCALE=42");
  if (extra_define) CeedAddJitDefine(*ceed, extra_define);
}

int main(int argc, char **argv) {
  Ceed ceed, ceed_other;

  // The same QFunctions, created again after the first ones are destroyed, in the same Ceed and in another Ceed with
  // the same JiT options: backends which reuse identical JiT compilations reuse those of the first QFunctions
  InitCeed(argv[1], NULL, &ceed);
  ApplyMass(ceed, 8);
  ApplyMass(ceed, 8);
  InitCeed(argv[1], NULL, &ceed_other);
  ApplyMass(ceed_other, 8);
  CeedDestroy(&ceed_other);

  // A Ceed with another JiT define compiles the QFunctions again
  InitCeed(argv[1], "UNUSED_DEFINE=1", &ceed_other);
  ApplyMass(ceed_other, 8);
  CeedDestroy(&ceed_other);

  // The QFunctions still work in the first Ceed after the other Ceeds are destroyed
  ApplyMass(ceed, 8);
  CeedDestroy(&ceed);
  return 0;
}
