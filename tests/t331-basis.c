/// @file
/// Test GetInterp and BasisApply for a 2D Quad non-tensor H(div) basis
/// \test Test GetInterp and BasisApply for a 2D Quad non-tensor H(div) basis
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#include "t330-basis.h"

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt num_nodes = 4, q = 3, dim = 2, num_qpts = q * q;
  CeedInt       num_comp = 1;                // one vector component
  CeedInt       p        = dim * num_nodes;  // DoF per element
  CeedBasis     basis;
  CeedScalar    q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar    div[p * num_qpts], interp[p * dim * num_qpts];
  CeedVector    u, v;

  CeedInit(argv[1], &ceed);

  BuildHdivQuadrilateral(q, q_ref, q_weights, interp, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp, p, num_qpts, interp, div, q_ref, q_weights, &basis);

  // Test GetInterp for H(div)
  {
    const CeedScalar *interp_2;
    CeedBasisGetInterp(basis, &interp_2);
    for (CeedInt i = 0; i < p * dim * num_qpts; i++) {
      if (fabs(interp[i] - interp_2[i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("%f != %f\n", interp[i], interp_2[i]);
        // LCOV_EXCL_STOP
      }
    }
  }

  CeedVectorCreate(ceed, p, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, num_qpts * dim, &v);
  CeedVectorSetValue(v, 0.);

  // BasisApply for H(div): CEED_EVAL_INTERP, NOTRANSPOSE case
  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < dim * num_qpts; i++) {
      if (fabs(q_ref[i] - v_array[i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("%f != %f\n", q_ref[i], v_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorSetValue(v, 1.0);
  CeedVectorSetValue(u, 0.0);

  // BasisApply for Hdiv: CEED_EVAL_INTERP, TRANSPOSE case
  CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, v, u);

  {
    const CeedScalar *u_array;

    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    CeedScalar sum = 0.;
    for (CeedInt i = 0; i < p; i++) {
      sum += u_array[i];
    }
    if (fabs(sum) > 100. * CEED_EPSILON) printf("sum of array %f != %f\n", sum, 0.0);
    CeedVectorRestoreArrayRead(u, &u_array);
  }

  CeedBasisDestroy(&basis);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
