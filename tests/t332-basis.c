/// @file
/// Test div with a 2D Quad non-tensor H(div) basis
/// \test Test div with a 2D Quad non-tensor H(div) basis
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#include "t330-basis.h"

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    u, v;
  const CeedInt p = 8, q = 3, dim = 2, num_qpts = q * q;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar    interp[dim * p * num_qpts], div[p * num_qpts];

  CeedInit(argv[1], &ceed);

  BuildHdivQuadrilateral(q, q_ref, q_weights, interp, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, 1, p, num_qpts, interp, div, q_ref, q_weights, &basis);

  // Test div for H(div)
  {
    const CeedScalar *div_in_basis;

    CeedBasisGetDiv(basis, &div_in_basis);
    for (CeedInt i = 0; i < p * num_qpts; i++) {
      if (fabs(div[i] - div_in_basis[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", div[i], div_in_basis[i]);
    }
  }

  CeedVectorCreate(ceed, p, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, num_qpts, &v);
  CeedVectorSetValue(v, 0.0);

  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_DIV, u, v);

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < num_qpts; i++) {
      if (fabs(p * 0.25 - v_array[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", 2.0, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorSetValue(v, 1.0);
  CeedVectorSetValue(u, 0.0);

  CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_DIV, v, u);

  {
    const CeedScalar *u_array;

    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    for (CeedInt i = 0; i < p; i++) {
      if (fabs(num_qpts * 0.25 - u_array[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", 2.0, u_array[i]);
    }
    CeedVectorRestoreArrayRead(u, &u_array);
  }

  CeedBasisDestroy(&basis);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
