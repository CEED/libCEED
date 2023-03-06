/// @file
/// Test creation, evaluation, and destruction for vector mass QFunction by name
/// \test Test creation, evaluation, and destruction for vector mass QFunction by name
#include <ceed.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    in[16], out[16];
  CeedVector    q_data, dx, w, u, v;
  CeedQFunction qf_setup, qf_mass;
  CeedInt       q        = 8;
  const CeedInt num_comp = 3;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 2; dim <= 3; dim++) {
    CeedInt num_qpts = CeedIntPow(q, dim);

    CeedVectorCreate(ceed, num_qpts * dim * dim, &dx);
    CeedVectorCreate(ceed, num_qpts, &w);
    CeedVectorCreate(ceed, num_qpts * num_comp, &u);
    {
      CeedScalar dx_array[num_qpts * dim * dim], w_array[num_qpts], u_array[num_qpts * num_comp];

      for (CeedInt i = 0; i < num_qpts; i++) {
        w_array[i] = 1.0 / num_qpts;
      }
      for (CeedInt d = 0; d < dim; d++) {
        for (CeedInt g = 0; g < dim; g++) {
          for (CeedInt i = 0; i < num_qpts; i++) {
            dx_array[i + (g * dim + d) * num_qpts] = d == g;
          }
        }
      }
      for (CeedInt c = 0; c < num_comp; c++) {
        for (CeedInt i = 0; i < num_qpts; i++) {
          u_array[i + c * num_qpts] = c + 1;
        }
      }
      CeedVectorSetArray(dx, CEED_MEM_HOST, CEED_COPY_VALUES, dx_array);
      CeedVectorSetArray(w, CEED_MEM_HOST, CEED_COPY_VALUES, w_array);
      CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
    }
    CeedVectorCreate(ceed, num_qpts, &q_data);
    CeedVectorSetValue(q_data, 0.0);
    CeedVectorCreate(ceed, num_qpts * num_comp, &v);
    CeedVectorSetValue(v, 0.0);

    char name[13] = "";
    snprintf(name, sizeof name, "Mass%" CeedInt_FMT "DBuild", dim);
    CeedQFunctionCreateInteriorByName(ceed, name, &qf_setup);
    {
      in[0]  = dx;
      in[1]  = w;
      out[0] = q_data;
      CeedQFunctionApply(qf_setup, num_qpts, in, out);
    }

    CeedQFunctionCreateInteriorByName(ceed, "Vector3MassApply", &qf_mass);
    {
      in[0]  = u;
      in[1]  = q_data;
      out[0] = v;
      CeedQFunctionApply(qf_mass, num_qpts, in, out);
    }

    // Verify results
    {
      const CeedScalar *v_array;

      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      for (CeedInt c = 0; c < num_comp; c++) {
        CeedScalar sum = 0;
        for (CeedInt i = 0; i < num_qpts; i++) sum += v_array[i + c * num_qpts];
        if (fabs(sum - (c + 1)) > 10 * CEED_EPSILON) {
          // LCOV_EXCL_START
          printf("%" CeedInt_FMT "D volume error in component %" CeedInt_FMT ": %f != %f\n", dim, c, sum, (c + 1.0));
          // LCOV_EXCL_STOP
        }
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
  }

  CeedDestroy(&ceed);
  return 0;
}
