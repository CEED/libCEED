/// @file
/// Compare independent basis coefficients and composite replay with the CPU oracle.
//TESTARGS(name="2D INTERP components") {ceed_resource} 2 3 4 8 7 0
//TESTARGS(name="2D padded gradient components") {ceed_resource} 2 6 4 8 7 1
//TESTARGS(name="3D INTERP tails") {ceed_resource} 3 3 4 2 7 0
//TESTARGS(name="3D GRAD tails and mixed Basis") {ceed_resource} 3 3 4 2 7 1
//TESTARGS(name="3D underintegrated gradient") {ceed_resource} 3 6 4 2 7 1
//TESTARGS(name="3D scalar collocated slices") {ceed_resource} 3 5 6 1 7 1 0
//TESTARGS(name="3D collocated gradient") {ceed_resource} 3 5 5 2 7 1
//TESTARGS(name="3D prefetched interpolation single element") {ceed_resource} 3 3 4 1 1 0

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "t527-operator.h"

static int CreateOperator(Ceed ceed, CeedInt dim, CeedInt p, CeedInt q, CeedInt num_comp, CeedInt num_elem, bool gradient, CeedScalar scale,
                          bool allow_mixed_basis, CeedOperator *op) {
  CeedBasis            base, basis;
  CeedElemRestriction  restriction;
  CeedQFunction        qf;
  CeedQFunctionContext context;
  const CeedScalar    *interp, *grad, *q_ref, *q_weight;
  CeedInt              nodes = 1, size = num_comp * (gradient ? dim : 1);

  for (CeedInt i = 0; i < dim; i++) nodes *= p;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p, q, CEED_GAUSS, &base);
  CeedBasisGetInterp1D(base, &interp);
  CeedBasisGetGrad1D(base, &grad);
  CeedBasisGetQRef(base, &q_ref);
  CeedBasisGetQWeights(base, &q_weight);
  CeedScalar scaled_interp[p * q], scaled_grad[p * q];

  for (CeedInt i = 0; i < p * q; i++) {
    scaled_interp[i] = p == q && scale == 1 ? (i % p == i / p) : scale * interp[i];
    scaled_grad[i]   = (scale + 0.1) * grad[i];
  }
  CeedBasisCreateTensorH1(ceed, dim, num_comp, p, q, scaled_interp, scaled_grad, q_ref, q_weight, &basis);
  CeedInt offsets[(num_elem ? num_elem : 1) * nodes];

  for (CeedInt e = 0; e < num_elem; e++) {
    for (CeedInt n = 0; n < nodes; n++) offsets[e * nodes + n] = e * (nodes - 1) + n;
  }

  CeedElemRestrictionCreate(ceed, num_elem, nodes, num_comp, num_elem ? nodes * num_elem : 1, nodes * num_comp * num_elem, CEED_MEM_HOST,
                            CEED_COPY_VALUES, offsets, &restriction);
  CeedQFunctionCreateInterior(ceed, 1, ScaleComponents, ScaleComponents_loc, &qf);
  CeedQFunctionAddInput(qf, "u", size, gradient ? CEED_EVAL_GRAD : CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf, "v", size, gradient ? CEED_EVAL_GRAD : CEED_EVAL_INTERP);
  const bool mixed_basis = allow_mixed_basis && dim == 3 && gradient && p < q;

  if (mixed_basis) CeedQFunctionAddInput(qf, "extra", 1, CEED_EVAL_INTERP);
  CeedQFunctionContextCreate(ceed, &context);
  PipelineContext context_data = {size, mixed_basis, 1};

  CeedQFunctionContextSetData(context, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof context_data, &context_data);
  CeedQFunctionSetContext(qf, context);
  CeedQFunctionSetContextWritable(qf, false);
  CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op);
  CeedOperatorSetField(*op, "u", restriction, basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*op, "v", restriction, basis, CEED_VECTOR_ACTIVE);
  if (mixed_basis) {
    // This larger passive Basis makes T>Q while the active gradient still uses
    // collocated slices. Collectives must include the padded x/y threads too
    CeedBasis           extra_basis;
    CeedElemRestriction extra_restriction;
    CeedVector          extra;
    const CeedInt       extra_p = q + 2, extra_nodes = extra_p * extra_p * extra_p;
    const CeedInt       extra_strides[3] = {1, extra_nodes, extra_nodes};

    CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, extra_p, q, CEED_GAUSS, &extra_basis);
    CeedElemRestrictionCreateStrided(ceed, num_elem, extra_nodes, 1, extra_nodes * num_elem, extra_strides, &extra_restriction);
    CeedVectorCreate(ceed, extra_nodes * num_elem, &extra);
    CeedVectorSetValue(extra, 0);
    CeedOperatorSetField(*op, "extra", extra_restriction, extra_basis, extra);
    CeedBasisDestroy(&extra_basis);
    CeedElemRestrictionDestroy(&extra_restriction);
    CeedVectorDestroy(&extra);
  }
  CeedBasisDestroy(&base);
  CeedBasisDestroy(&basis);
  CeedElemRestrictionDestroy(&restriction);
  CeedQFunctionDestroy(&qf);
  CeedQFunctionContextDestroy(&context);
  return CEED_ERROR_SUCCESS;
}

static int CompareVectors(CeedVector actual, CeedVector expected, const char *phase) {
  const CeedScalar *actual_arr, *expected_arr;
  CeedScalar        scale = 1;
  CeedSize          length;

  CeedVectorGetLength(actual, &length);
  CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &actual_arr);
  CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &expected_arr);
  for (CeedSize i = 0; i < length; i++) scale = fmax(scale, fabs(expected_arr[i]));
  for (CeedSize i = 0; i < length; i++) {
    const CeedScalar error = fabs(actual_arr[i] - expected_arr[i]) / scale;

    if (!(error <= 500 * CEED_EPSILON)) {
      printf("%s mismatch at %td: %g != %g\n", phase, i, actual_arr[i], expected_arr[i]);
      break;
    }
  }
  CeedVectorRestoreArrayRead(actual, &actual_arr);
  CeedVectorRestoreArrayRead(expected, &expected_arr);
  return CEED_ERROR_SUCCESS;
}

int main(int argc, char **argv) {
  Ceed     ceed, ceed_ref;
  CeedInt  dim = atoi(argv[2]), p = atoi(argv[3]), q = atoi(argv[4]), num_comp = atoi(argv[5]), num_elem = atoi(argv[6]);
  bool     gradient = atoi(argv[7]);
  CeedSize length   = num_elem * num_comp;

  for (CeedInt i = 0; i < dim; i++) length *= p;
  CeedOperator op[2], op_ref[2], op_composite, op_composite_ref;
  CeedVector   u, v, u_ref, v_ref;

  CeedInit(argv[1], &ceed);
  CeedInit("/cpu/self/ref/serial", &ceed_ref);
  CeedVectorCreate(ceed, length, &u);
  CeedVectorCreate(ceed, length, &v);
  CeedVectorCreate(ceed_ref, length, &u_ref);
  CeedVectorCreate(ceed_ref, length, &v_ref);
  CeedVector inputs[2] = {u, u_ref};

  for (CeedInt side = 0; side < 2; side++) {
    CeedScalar *input_arr;

    CeedVectorGetArrayWrite(inputs[side], CEED_MEM_HOST, &input_arr);
    for (CeedSize i = 0; i < length; i++) input_arr[i] = sin(0.017 * i) + 0.2 * cos(0.037 * i);
    CeedVectorRestoreArray(inputs[side], &input_arr);
  }
  const bool allow_mixed_basis = argc <= 8 || atoi(argv[8]);

  for (CeedInt i = 0; i < 2; i++) {
    CreateOperator(ceed, dim, p, q, num_comp, num_elem, gradient, 1 + 0.2 * i, allow_mixed_basis, &op[i]);
    CreateOperator(ceed_ref, dim, p, q, num_comp, num_elem, gradient, 1 + 0.2 * i, allow_mixed_basis, &op_ref[i]);
  }
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCreateComposite(ceed_ref, &op_composite_ref);
  for (CeedInt i = 0; i < 2; i++) {
    CeedOperatorCompositeAddSub(op_composite, op[i]);
    CeedOperatorCompositeAddSub(op_composite_ref, op_ref[i]);
  }
  // Alternate distinct same-shape modules, then capture and replay their composite
  for (CeedInt step = 0; step < 8; step++) {
    if (step >= 5) {
      for (CeedInt i = 0; i < 2; i++) {
        CeedOperator pair[2] = {op[i], op_ref[i]};

        for (CeedInt side = 0; side < 2; side++) {
          CeedQFunctionContext context;
          PipelineContext     *values;

          CeedOperatorGetContext(pair[side], &context);
          CeedQFunctionContextGetData(context, CEED_MEM_HOST, &values);
          values->scale = 1 + 0.05 * step;
          CeedQFunctionContextRestoreData(context, &values);
          CeedQFunctionContextDestroy(&context);
          if (allow_mixed_basis && dim == 3 && gradient && p < q) {
            CeedOperatorField field;
            CeedVector        extra;

            CeedOperatorGetFieldByName(pair[side], "extra", &field);
            CeedOperatorFieldGetVector(field, &extra);
            CeedVectorSetValue(extra, 0.2 * step);
            CeedVectorDestroy(&extra);
          }
        }
      }
    }
    CeedOperator actual   = step < 4 ? op[step % 2] : op_composite;
    CeedOperator expected = step < 4 ? op_ref[step % 2] : op_composite_ref;

    CeedOperatorApply(actual, u, v, CEED_REQUEST_IMMEDIATE);
    CeedOperatorApply(expected, u_ref, v_ref, CEED_REQUEST_IMMEDIATE);
    CompareVectors(v, v_ref, "Apply and composite replay");
  }
  // Passive coefficients must preserve linearity in the active vector
  if (allow_mixed_basis && dim == 3 && gradient && p < q) {
    CeedScalar norm;

    CeedVectorSetValue(u, 0);
    CeedOperatorApply(op[0], u, v, CEED_REQUEST_IMMEDIATE);
    CeedVectorNorm(v, CEED_NORM_MAX, &norm);
    if (!(norm <= 500 * CEED_EPSILON)) {
      printf("Nonzero output for zero active input: %g\n", norm);
    }
    CeedVectorCopy(u_ref, u);
  }
  // Check diagonal assembly against the same independent oracle
  if (num_elem <= 17) {
    CeedOperatorLinearAssembleDiagonal(op[0], v, CEED_REQUEST_IMMEDIATE);
    CeedOperatorLinearAssembleDiagonal(op_ref[0], v_ref, CEED_REQUEST_IMMEDIATE);
    CompareVectors(v, v_ref, "Diagonal assembly");
  }
  // Vectors outlive the operators; check that destruction preserves their results
  CeedOperatorApply(op_composite, u, v, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_composite_ref, u_ref, v_ref, CEED_REQUEST_IMMEDIATE);
  CeedOperatorDestroy(&op_composite);
  CeedOperatorDestroy(&op_composite_ref);
  for (CeedInt i = 0; i < 2; i++) {
    CeedOperatorDestroy(&op[i]);
    CeedOperatorDestroy(&op_ref[i]);
  }
  CompareVectors(v, v_ref, "After operator destruction");
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&u_ref);
  CeedVectorDestroy(&v_ref);
  CeedDestroy(&ceed);
  CeedDestroy(&ceed_ref);
  return CEED_ERROR_SUCCESS;
}
