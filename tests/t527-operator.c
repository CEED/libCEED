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
  CeedScalar *scaled_interp = malloc(p * q * sizeof *scaled_interp), *scaled_grad = malloc(p * q * sizeof *scaled_grad);

  if (!scaled_interp || !scaled_grad) {
    free(scaled_interp);
    free(scaled_grad);
    return CEED_ERROR_MAJOR;
  }
  for (CeedInt i = 0; i < p * q; i++) {
    scaled_interp[i] = p == q && scale == 1 ? (i % p == i / p) : scale * interp[i];
    scaled_grad[i]   = (scale + 0.1) * grad[i];
  }
  const int basis_error = CeedBasisCreateTensorH1(ceed, dim, num_comp, p, q, scaled_interp, scaled_grad, q_ref, q_weight, &basis);
  free(scaled_interp);
  free(scaled_grad);
  if (basis_error) return 1;
  CeedInt *offsets = malloc(num_elem * nodes * sizeof(CeedInt));

  if (num_elem && !offsets) return CEED_ERROR_MAJOR;

  for (CeedInt e = 0; e < num_elem; e++) {
    for (CeedInt n = 0; n < nodes; n++) offsets[e * nodes + n] = e * (nodes - 1) + n;
  }

  const int restriction_error = CeedElemRestrictionCreate(ceed, num_elem, nodes, num_comp, num_elem ? nodes * num_elem : 1,
                                                          nodes * num_comp * num_elem, CEED_MEM_HOST, CEED_COPY_VALUES, offsets, &restriction);
  free(offsets);
  if (restriction_error) return 1;
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
    // collocated slices. Collectives must include the padded x/y threads too.
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
  return 0;
}

static int CompareVectors(CeedVector actual, CeedVector expected, CeedSize length, const char *phase) {
  const CeedScalar *result, *oracle;
  CeedScalar        max_error = 0, max_value = 0;
  bool              finite = true;

  CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &result);
  CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &oracle);
  for (CeedSize i = 0; i < length; i++) {
    finite    = finite && isfinite(result[i]) && isfinite(oracle[i]);
    max_error = fmax(max_error, fabs(result[i] - oracle[i]));
    max_value = fmax(max_value, fabs(oracle[i]));
  }
  CeedVectorRestoreArrayRead(actual, &result);
  CeedVectorRestoreArrayRead(expected, &oracle);
  if (!finite || max_error > 500 * CEED_EPSILON * fmax(1, max_value)) {
    printf("%s: error %g, reference scale %g, finite %d\n", phase, max_error, max_value, finite);
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 8) return 1;
  Ceed     ceed, reference;
  CeedInt  dim = atoi(argv[2]), p = atoi(argv[3]), q = atoi(argv[4]), num_comp = atoi(argv[5]), num_elem = atoi(argv[6]);
  bool     gradient = atoi(argv[7]);
  CeedSize length   = num_elem * num_comp;

  for (CeedInt i = 0; i < dim; i++) length *= p;
  CeedOperator operators[2], oracles[2], composite, reference_composite;
  CeedVector   u, v, u_ref, v_ref;

  CeedInit(argv[1], &ceed);
  CeedInit("/cpu/self/ref/serial", &reference);
  CeedVectorCreate(ceed, length, &u);
  CeedVectorCreate(ceed, length, &v);
  CeedVectorCreate(reference, length, &u_ref);
  CeedVectorCreate(reference, length, &v_ref);
  CeedVector inputs[2] = {u, u_ref};

  for (CeedInt side = 0; side < 2; side++) {
    CeedScalar *input;

    CeedVectorGetArrayWrite(inputs[side], CEED_MEM_HOST, &input);
    for (CeedSize i = 0; i < length; i++) input[i] = sin(0.017 * i) + 0.2 * cos(0.037 * i);
    CeedVectorRestoreArray(inputs[side], &input);
  }
  const bool allow_mixed_basis = argc <= 8 || atoi(argv[8]);

  for (CeedInt i = 0; i < 2; i++) {
    if (CreateOperator(ceed, dim, p, q, num_comp, num_elem, gradient, 1 + 0.2 * i, allow_mixed_basis, &operators[i])) return 1;
    if (CreateOperator(reference, dim, p, q, num_comp, num_elem, gradient, 1 + 0.2 * i, allow_mixed_basis, &oracles[i])) return 1;
  }
  CeedOperatorCreateComposite(ceed, &composite);
  CeedOperatorCreateComposite(reference, &reference_composite);
  for (CeedInt i = 0; i < 2; i++) {
    CeedOperatorCompositeAddSub(composite, operators[i]);
    CeedOperatorCompositeAddSub(reference_composite, oracles[i]);
  }
  // Alternate distinct same-shape modules, then capture and replay their composite.
  for (CeedInt step = 0; step < 8; step++) {
    if (step >= 5) {
      for (CeedInt i = 0; i < 2; i++) {
        CeedOperator pair[2] = {operators[i], oracles[i]};

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
    CeedOperator actual   = step < 4 ? operators[step % 2] : composite;
    CeedOperator expected = step < 4 ? oracles[step % 2] : reference_composite;

    CeedOperatorApply(actual, u, v, CEED_REQUEST_IMMEDIATE);
    CeedOperatorApply(expected, u_ref, v_ref, CEED_REQUEST_IMMEDIATE);
    if (CompareVectors(v, v_ref, length, "Apply and composite replay")) return 1;
  }
  // Passive coefficients must preserve linearity in the active vector.
  if (allow_mixed_basis && dim == 3 && gradient && p < q) {
    CeedScalar norm;

    CeedVectorSetValue(u, 0);
    CeedOperatorApply(operators[0], u, v, CEED_REQUEST_IMMEDIATE);
    CeedVectorNorm(v, CEED_NORM_MAX, &norm);
    if (!isfinite(norm) || norm > 500 * CEED_EPSILON) {
      printf("Nonzero output for zero active input: %g\n", norm);
      return 1;
    }
    CeedVectorCopy(u_ref, u);
  }
  // Check diagonal assembly against the same independent oracle.
  if (num_elem <= 17) {
    CeedOperatorLinearAssembleDiagonal(operators[0], v, CEED_REQUEST_IMMEDIATE);
    CeedOperatorLinearAssembleDiagonal(oracles[0], v_ref, CEED_REQUEST_IMMEDIATE);
    if (CompareVectors(v, v_ref, length, "Diagonal assembly")) return 1;
  }
  // Vectors outlive the operators; check that destruction preserves their results.
  CeedOperatorApply(composite, u, v, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(reference_composite, u_ref, v_ref, CEED_REQUEST_IMMEDIATE);
  CeedOperatorDestroy(&composite);
  CeedOperatorDestroy(&reference_composite);
  for (CeedInt i = 0; i < 2; i++) {
    CeedOperatorDestroy(&operators[i]);
    CeedOperatorDestroy(&oracles[i]);
  }
  if (CompareVectors(v, v_ref, length, "After operator destruction")) return 1;
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&u_ref);
  CeedVectorDestroy(&v_ref);
  CeedDestroy(&ceed);
  CeedDestroy(&reference);
  return 0;
}
