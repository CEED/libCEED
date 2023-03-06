/// @file
/// Test creation, evaluation, and destruction for QFunction
/// \test Test creation, evaluation, and destruction for QFunction
#include "t400-qfunction.h"

#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    in[16], out[16];
  CeedVector    q_data, w, u, v;
  CeedQFunction qf_setup, qf_mass;
  CeedInt       q = 8;
  CeedScalar    v_true[q];

  CeedInit(argv[1], &ceed);

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
    in[0]  = w;
    in[1]  = u;
    out[0] = v;
    CeedQFunctionApply(qf_mass, q, in, out);
  }

  // Verify result
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < q; i++) {
      if (v_array[i] != v_true[i]) printf("[%" CeedInt_FMT "] v %f != v_true %f\n", i, v_array[i], v_true[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&w);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
