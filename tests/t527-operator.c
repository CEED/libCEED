/// @file
/// Test composite operator with a passive output set to a different array before each apply
/// \test Test composite operator with a passive output set to a different array before each apply
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t590-operator.h"

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_u, elem_restriction_q;
  CeedBasis           basis_u;
  CeedQFunction       qf_mass;
  CeedOperator        op_mass, op_composite;
  CeedVector          u, v, v_arrays[4];
  CeedMemType         mem_type;
  CeedInt             num_elem = 3, p = 2, q = 3, num_dofs = num_elem * (p - 1) + 1, num_qpts = num_elem * q, num_applies = 4;
  CeedInt             ind_u[num_elem * p];

  CeedInit(argv[1], &ceed);
  CeedGetPreferredMemType(ceed, &mem_type);

  // Vectors
  CeedVectorCreate(ceed, num_dofs, &u);
  CeedVectorCreate(ceed, num_qpts, &v);
  for (CeedInt k = 0; k < num_applies; k++) {
    CeedVectorCreate(ceed, num_qpts, &v_arrays[k]);
    CeedVectorSetValue(v_arrays[k], 0.0);
  }

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p; j++) ind_u[p * i + j] = i * (p - 1) + j;
  }
  CeedElemRestrictionCreate(ceed, num_elem, p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_COPY_VALUES, ind_u, &elem_restriction_u);
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, num_qpts, CEED_STRIDES_BACKEND, &elem_restriction_q);

  // Basis
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis_u);

  // QFunction
  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_NONE);

  // Operators, with an active input and a passive output
  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_q, CEED_BASIS_NONE, v);
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCompositeAddSub(op_composite, op_mass);

  // Apply, lending the passive output a different array each time, as a caller writing into its own vectors would
  for (CeedInt k = 0; k < num_applies; k++) {
    CeedScalar *array;

    CeedVectorSetValue(u, k + 1.0);
    CeedVectorGetArrayWrite(v_arrays[k], mem_type, &array);
    CeedVectorSetArray(v, mem_type, CEED_USE_POINTER, array);
    CeedOperatorApply(op_composite, u, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE);
    CeedVectorTakeArray(v, mem_type, &array);
    CeedVectorRestoreArray(v_arrays[k], &array);
  }

  // Check that each apply wrote into its own array
  for (CeedInt k = 0; k < num_applies; k++) {
    const CeedScalar *array;

    CeedVectorGetArrayRead(v_arrays[k], CEED_MEM_HOST, &array);
    for (CeedInt i = 0; i < num_qpts; i++) {
      if (fabs(array[i] - (k + 1.0)) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Apply %" CeedInt_FMT ", passive output entry %" CeedInt_FMT ": %f != %f\n", k, i, array[i], k + 1.0);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(v_arrays[k], &array);
  }

  // Cleanup
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  for (CeedInt k = 0; k < num_applies; k++) CeedVectorDestroy(&v_arrays[k]);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_q);
  CeedBasisDestroy(&basis_u);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_composite);
  CeedDestroy(&ceed);
  return 0;
}
