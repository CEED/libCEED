/// @file
/// Test creation, evaluation, and destruction for vector mass QFunction by name
/// \test Test creation, evaluation, and destruction for vector mass QFunction by name
#include <ceed.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        in[16], out[16];
  CeedVector        Q_data, J, W, U, V;
  CeedQFunction     qf_setup, qf_mass;
  CeedInt           Q        = 8;
  const CeedInt     num_comp = 3;
  const CeedScalar *vv;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 2; dim <= 3; dim++) {
    CeedInt    num_qpts = CeedIntPow(Q, dim);
    CeedScalar j[num_qpts * dim * dim], w[num_qpts], u[num_qpts * num_comp];

    char name[13] = "";
    snprintf(name, sizeof name, "Mass%" CeedInt_FMT "DBuild", dim);
    CeedQFunctionCreateInteriorByName(ceed, name, &qf_setup);
    CeedQFunctionCreateInteriorByName(ceed, "Vector3MassApply", &qf_mass);

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
      for (CeedInt i = 0; i < num_qpts; i++) {
        u[i + c * num_qpts] = c + 1;
      }
    }

    CeedVectorCreate(ceed, num_qpts * dim * dim, &J);
    CeedVectorSetArray(J, CEED_MEM_HOST, CEED_USE_POINTER, j);
    CeedVectorCreate(ceed, num_qpts, &W);
    CeedVectorSetArray(W, CEED_MEM_HOST, CEED_USE_POINTER, w);
    CeedVectorCreate(ceed, num_qpts, &Q_data);
    CeedVectorSetValue(Q_data, 0.0);
    CeedVectorCreate(ceed, num_qpts * num_comp, &U);
    CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);
    CeedVectorCreate(ceed, num_qpts * num_comp, &V);
    CeedVectorSetValue(V, 0.0);

    {
      in[0]  = J;
      in[1]  = W;
      out[0] = Q_data;
      CeedQFunctionApply(qf_setup, num_qpts, in, out);
    }
    {
      in[0]  = U;
      in[1]  = Q_data;
      out[0] = V;
      CeedQFunctionApply(qf_mass, num_qpts, in, out);
    }

    CeedVectorGetArrayRead(V, CEED_MEM_HOST, &vv);
    for (CeedInt c = 0; c < num_comp; c++) {
      CeedScalar sum = 0;
      for (CeedInt i = 0; i < num_qpts; i++) sum += vv[i + c * num_qpts];
      if (fabs(sum - (c + 1)) > 10 * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("%" CeedInt_FMT "D volume error in component %" CeedInt_FMT ": %f != %f\n", dim, c, sum, (c + 1.0));
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(V, &vv);

    CeedVectorDestroy(&J);
    CeedVectorDestroy(&W);
    CeedVectorDestroy(&U);
    CeedVectorDestroy(&V);
    CeedVectorDestroy(&Q_data);
    CeedQFunctionDestroy(&qf_setup);
    CeedQFunctionDestroy(&qf_mass);
  }

  CeedDestroy(&ceed);
  return 0;
}
