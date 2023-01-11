/// @file
/// Test creation, evaluation, and destruction for vector Poisson QFunction by name
/// \test Test creation, evaluation, and destruction for vector Poisson QFunction by name
#include <ceed.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        in[16], out[16];
  CeedVector        Q_data, J, W, dU, dV;
  CeedQFunction     qf_setup, qf_diff;
  CeedInt           Q        = 8;
  const CeedInt     num_comp = 3;
  const CeedScalar *vv;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedInt    num_qpts = CeedIntPow(Q, dim);
    CeedScalar j[num_qpts * dim * dim], w[num_qpts], du[num_qpts * dim * num_comp];

    char name_setup[26] = "", name_apply[26] = "";
    snprintf(name_setup, sizeof name_setup, "Poisson%" CeedInt_FMT "DBuild", dim);
    CeedQFunctionCreateInteriorByName(ceed, name_setup, &qf_setup);
    snprintf(name_apply, sizeof name_apply, "Vector3Poisson%" CeedInt_FMT "DApply", dim);
    CeedQFunctionCreateInteriorByName(ceed, name_apply, &qf_diff);

    for (CeedInt i = 0; i < num_qpts; i++) {
      w[i] = 1.0 / num_qpts;
    }
    for (CeedInt d = 0; d < dim; d++) {
      for (CeedInt g = 0; g < dim; g++) {
        for (CeedInt i = 0; i < num_qpts; i++) {
          j[i + (g * dim + d) * num_qpts] = d == g;
        }
      }
    }
    for (CeedInt c = 0; c < num_comp; c++) {
      for (CeedInt g = 0; g < dim; g++) {
        for (CeedInt i = 0; i < num_qpts; i++) {
          du[i + (g * num_comp + c) * num_qpts] = c + 1;
        }
      }
    }

    CeedVectorCreate(ceed, num_qpts * dim * dim, &J);
    CeedVectorSetArray(J, CEED_MEM_HOST, CEED_USE_POINTER, j);
    CeedVectorCreate(ceed, num_qpts, &W);
    CeedVectorSetArray(W, CEED_MEM_HOST, CEED_USE_POINTER, w);
    CeedVectorCreate(ceed, num_qpts * dim * (dim + 1) / 2, &Q_data);
    CeedVectorSetValue(Q_data, 0.0);
    CeedVectorCreate(ceed, num_qpts * dim * num_comp, &dU);
    CeedVectorSetArray(dU, CEED_MEM_HOST, CEED_USE_POINTER, du);
    CeedVectorCreate(ceed, num_qpts * dim * num_comp, &dV);
    CeedVectorSetValue(dV, 0.0);

    {
      in[0]  = J;
      in[1]  = W;
      out[0] = Q_data;
      CeedQFunctionApply(qf_setup, num_qpts, in, out);
    }
    {
      in[0]  = dU;
      in[1]  = Q_data;
      out[0] = dV;
      CeedQFunctionApply(qf_diff, num_qpts, in, out);
    }

    CeedVectorGetArrayRead(dV, CEED_MEM_HOST, &vv);
    for (CeedInt c = 0; c < num_comp; c++) {
      CeedScalar sum = 0;
      for (CeedInt i = 0; i < num_qpts; i++) {
        for (CeedInt g = 0; g < dim; g++) sum += vv[i + (g * num_comp + c) * num_qpts];
      }
      if (fabs(sum - dim * (c + 1)) > 10 * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("%" CeedInt_FMT "D volume error in component %" CeedInt_FMT ": %f != %f\n", dim, c, sum, dim * (c + 1.0));
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(dV, &vv);

    CeedVectorDestroy(&J);
    CeedVectorDestroy(&W);
    CeedVectorDestroy(&dU);
    CeedVectorDestroy(&dV);
    CeedVectorDestroy(&Q_data);
    CeedQFunctionDestroy(&qf_setup);
    CeedQFunctionDestroy(&qf_diff);
  }

  CeedDestroy(&ceed);
  return 0;
}
