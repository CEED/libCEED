/// @file
/// Test creation, evaluation, and destruction for vector Poisson QFunction by name
/// \test Test creation, evaluation, and destruction for vector Poisson QFunction by name
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    in[16], out[16];
  CeedVector    q_data, dx, w, du, dv;
  CeedQFunction qf_setup, qf_diff;
  CeedInt       q        = 8;
  const CeedInt num_comp = 3;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedInt num_qpts = CeedIntPow(q, dim);

    CeedVectorCreate(ceed, num_qpts * dim * dim, &dx);
    CeedVectorCreate(ceed, num_qpts, &w);
    CeedVectorCreate(ceed, num_qpts * dim * num_comp, &du);
    {
      CeedScalar dx_array[num_qpts * dim * dim], w_array[num_qpts], du_array[num_qpts * dim * num_comp];

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
        for (CeedInt g = 0; g < dim; g++) {
          for (CeedInt i = 0; i < num_qpts; i++) {
            du_array[i + (g * num_comp + c) * num_qpts] = c + 1;
          }
        }
      }

      CeedVectorSetArray(dx, CEED_MEM_HOST, CEED_COPY_VALUES, dx_array);
      CeedVectorSetArray(w, CEED_MEM_HOST, CEED_COPY_VALUES, w_array);
      CeedVectorSetArray(du, CEED_MEM_HOST, CEED_COPY_VALUES, du_array);
    }
    CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &q_data);
    CeedVectorSetValue(q_data, 0.0);
    CeedVectorCreate(ceed, num_qpts * dim * num_comp, &dv);
    CeedVectorSetValue(dv, 0.0);

    char name_setup[26] = "", name_apply[26] = "";
    snprintf(name_setup, sizeof name_setup, "Poisson%" CeedInt_FMT "DBuild", dim);
    CeedQFunctionCreateInteriorByName(ceed, name_setup, &qf_setup);
    {
      in[0]  = dx;
      in[1]  = w;
      out[0] = q_data;
      CeedQFunctionApply(qf_setup, num_qpts, in, out);
    }

    snprintf(name_apply, sizeof name_apply, "Vector3Poisson%" CeedInt_FMT "DApply", dim);
    CeedQFunctionCreateInteriorByName(ceed, name_apply, &qf_diff);
    {
      in[0]  = du;
      in[1]  = q_data;
      out[0] = dv;
      CeedQFunctionApply(qf_diff, num_qpts, in, out);
    }

    // Verify results
    {
      const CeedScalar *v_array;

      CeedVectorGetArrayRead(dv, CEED_MEM_HOST, &v_array);
      for (CeedInt c = 0; c < num_comp; c++) {
        CeedScalar sum = 0;
        for (CeedInt i = 0; i < num_qpts; i++) {
          for (CeedInt g = 0; g < dim; g++) sum += v_array[i + (g * num_comp + c) * num_qpts];
        }
        if (fabs(sum - dim * (c + 1)) > 10 * CEED_EPSILON) {
          // LCOV_EXCL_START
          printf("%" CeedInt_FMT "D volume error in component %" CeedInt_FMT ": %f != %f\n", dim, c, sum, dim * (c + 1.0));
          // LCOV_EXCL_STOP
        }
      }
      CeedVectorRestoreArrayRead(dv, &v_array);
    }

    CeedVectorDestroy(&dx);
    CeedVectorDestroy(&w);
    CeedVectorDestroy(&du);
    CeedVectorDestroy(&dv);
    CeedVectorDestroy(&q_data);
    CeedQFunctionDestroy(&qf_setup);
    CeedQFunctionDestroy(&qf_diff);
  }

  CeedDestroy(&ceed);
  return 0;
}
