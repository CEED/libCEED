/// @file
/// Test creation, evaluation, and destruction for QFunction by name
/// \test Test creation, evaluation, and destruction for QFunction by name
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    in[16], out[16];
  CeedVector    q_data, dx, w, u, v;
  CeedQFunction qf_setup, qf_mass;
  CeedInt       q = 8;
  CeedScalar    v_true[q];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, q, &dx);
  CeedVectorCreate(ceed, q, &w);
  CeedVectorCreate(ceed, q, &u);
  {
    CeedScalar dx_array[q], w_array[q], u_array[q];

    for (CeedInt i = 0; i < q; i++) {
      CeedScalar x = 2. * i / (q - 1) - 1;
      dx_array[i]  = 1;
      w_array[i]   = 1 - x * x;
      u_array[i]   = 2 + 3 * x + 5 * x * x;
      v_true[i]    = w_array[i] * u_array[i];
    }
    CeedVectorSetArray(dx, CEED_MEM_HOST, CEED_COPY_VALUES, dx_array);
    CeedVectorSetArray(w, CEED_MEM_HOST, CEED_COPY_VALUES, w_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }
  CeedVectorCreate(ceed, q, &v);
  CeedVectorSetValue(v, 0);
  CeedVectorCreate(ceed, q, &q_data);
  CeedVectorSetValue(q_data, 0);

  CeedQFunctionCreateInteriorByName(ceed, "Mass1DBuild", &qf_setup);
  {
    in[0]  = dx;
    in[1]  = w;
    out[0] = q_data;
    CeedQFunctionApply(qf_setup, q, in, out);
  }

  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);
  {
    in[0]  = w;
    in[1]  = u;
    out[0] = v;
    CeedQFunctionApply(qf_mass, q, in, out);
  }

  // Verify results
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < q; i++) {
      if (v_true[i] != v_array[i]) printf("[%" CeedInt_FMT "] v_true %f != v %f\n", i, v_true[i], v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&dx);
  CeedVectorDestroy(&w);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
