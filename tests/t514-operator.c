/// @file
/// Test CeedOperatorApply for composite operators with suboperators of different component layouts, and with a suboperator without elements that has a passive output
/// \test Test CeedOperatorApply for composite operators with suboperators of different component layouts, and with a suboperator without elements that has a passive output
#include "t514-operator.h"

#include <ceed.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_a, elem_restriction_b, elem_restriction_c, elem_restriction_empty;
  CeedQFunction       qf_ones_1, qf_ones_2;
  CeedOperator        op_a, op_b, op_c, op_empty, op_composite;
  CeedVector          u, v, p;
  CeedInt             offsets_0[1] = {0}, offsets_1[1] = {1};

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &u);
  CeedVectorCreate(ceed, 2, &v);
  CeedVectorCreate(ceed, 2, &p);
  CeedVectorSetValue(u, 0.0);

  // Restrictions of one node: one component at 0, two components at 0 and 1, one component at 1, and one without elements
  CeedElemRestrictionCreate(ceed, 1, 1, 1, 1, 2, CEED_MEM_HOST, CEED_COPY_VALUES, offsets_0, &elem_restriction_a);
  CeedElemRestrictionCreate(ceed, 1, 1, 2, 1, 2, CEED_MEM_HOST, CEED_COPY_VALUES, offsets_0, &elem_restriction_b);
  CeedElemRestrictionCreate(ceed, 1, 1, 1, 1, 2, CEED_MEM_HOST, CEED_COPY_VALUES, offsets_1, &elem_restriction_c);
  CeedElemRestrictionCreate(ceed, 0, 1, 1, 1, 2, CEED_MEM_HOST, CEED_COPY_VALUES, offsets_0, &elem_restriction_empty);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, ones_1, ones_1_loc, &qf_ones_1);
  CeedQFunctionAddInput(qf_ones_1, "u", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ones_1, "v", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, ones_2, ones_2_loc, &qf_ones_2);
  CeedQFunctionAddInput(qf_ones_2, "u", 2, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_ones_2, "v", 2, CEED_EVAL_NONE);

  // Operators writing ones
  CeedOperatorCreate(ceed, qf_ones_1, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_a);
  CeedOperatorSetField(op_a, "u", elem_restriction_a, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_a, "v", elem_restriction_a, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_ones_2, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_b);
  CeedOperatorSetField(op_b, "u", elem_restriction_b, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_b, "v", elem_restriction_b, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_ones_1, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_c);
  CeedOperatorSetField(op_c, "u", elem_restriction_c, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_c, "v", elem_restriction_c, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_ones_1, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_empty);
  CeedOperatorSetField(op_empty, "u", elem_restriction_empty, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_empty, "v", elem_restriction_empty, CEED_BASIS_NONE, p);

  // Suboperators with different component layouts writing the same entries
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCompositeAddSub(op_composite, op_a);
  CeedOperatorCompositeAddSub(op_composite, op_b);
  CeedOperatorCompositeAddSub(op_composite, op_c);
  CeedVectorSetValue(v, 5.0);
  CeedOperatorApply(op_composite, u, v, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < 2; i++) {
      if (v_array[i] != 2.0) printf("Layouts: v[%" CeedInt_FMT "] = %f != 2.0\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }
  CeedOperatorDestroy(&op_composite);

  // Suboperator without elements, whose passive output Apply still zeroes
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCompositeAddSub(op_composite, op_a);
  CeedOperatorCompositeAddSub(op_composite, op_empty);
  CeedVectorSetValue(v, 5.0);
  CeedVectorSetValue(p, 7.0);
  CeedOperatorApply(op_composite, u, v, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *v_array, *p_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    CeedVectorGetArrayRead(p, CEED_MEM_HOST, &p_array);
    if (v_array[0] != 1.0 || v_array[1] != 0.0) printf("Empty suboperator: v = [%f, %f] != [1.0, 0.0]\n", v_array[0], v_array[1]);
    if (p_array[0] != 0.0 || p_array[1] != 0.0) printf("Empty suboperator: p = [%f, %f] != [0.0, 0.0]\n", p_array[0], p_array[1]);
    CeedVectorRestoreArrayRead(v, &v_array);
    CeedVectorRestoreArrayRead(p, &p_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&p);
  CeedElemRestrictionDestroy(&elem_restriction_a);
  CeedElemRestrictionDestroy(&elem_restriction_b);
  CeedElemRestrictionDestroy(&elem_restriction_c);
  CeedElemRestrictionDestroy(&elem_restriction_empty);
  CeedQFunctionDestroy(&qf_ones_1);
  CeedQFunctionDestroy(&qf_ones_2);
  CeedOperatorDestroy(&op_a);
  CeedOperatorDestroy(&op_b);
  CeedOperatorDestroy(&op_c);
  CeedOperatorDestroy(&op_empty);
  CeedOperatorDestroy(&op_composite);
  CeedDestroy(&ceed);
  return 0;
}
