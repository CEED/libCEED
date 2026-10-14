/// @file
/// Compare independent basis coefficients and composite replay with the CPU oracle.
//TESTARGS(name="1D INTERP") {ceed_resource} 1 3 4 2 7 0
//TESTARGS(name="1D element packing tails") {ceed_resource} 1 2 2 1 257 0
//TESTARGS(name="2D GRAD") {ceed_resource} 2 3 4 2 7 1
//TESTARGS(name="1D GRAD components") {ceed_resource} 1 3 4 8 7 1
//TESTARGS(name="1D empty components") {ceed_resource} 1 3 4 2 0 0
//TESTARGS(name="2D INTERP components") {ceed_resource} 2 3 4 8 7 0
//TESTARGS(name="2D underintegration") {ceed_resource} 2 5 3 2 7 1
//TESTARGS(name="2D padded gradient components") {ceed_resource} 2 6 4 8 7 1
//TESTARGS(name="2D collocated components") {ceed_resource} 2 4 4 2 7 1
//TESTARGS(name="2D empty components") {ceed_resource} 2 3 4 2 0 0
//TESTARGS(name="2D single element") {ceed_resource} 2 3 4 2 1 0
//TESTARGS(name="3D INTERP tails") {ceed_resource} 3 3 4 2 7 0
//TESTARGS(name="3D GRAD tails and mixed Basis") {ceed_resource} 3 3 4 2 7 1
//TESTARGS(name="3D multiple components") {ceed_resource} 3 5 6 8 17 0
//TESTARGS(name="3D underintegration") {ceed_resource} 3 6 4 2 7 0
//TESTARGS(name="3D underintegrated gradient") {ceed_resource} 3 6 4 2 7 1
//TESTARGS(name="3D large basis") {ceed_resource} 3 9 10 1 7 0
//TESTARGS(name="3D collocated interpolation") {ceed_resource} 3 5 5 2 7 0
//TESTARGS(name="3D collocated gradient") {ceed_resource} 3 5 5 2 7 1
//TESTARGS(name="3D scalar collocated slices") {ceed_resource} 3 5 6 1 7 1 0 0
//TESTARGS(name="3D empty operator") {ceed_resource} 3 3 4 1 0 0
//TESTARGS(name="3D single element") {ceed_resource} 3 3 4 1 1 0

#define _POSIX_C_SOURCE 200809L
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "t527-operator.h"

static int CreateOperator(Ceed ceed, CeedInt dim, CeedInt p, CeedInt q, CeedInt num_comp, CeedInt num_elem, bool gradient, CeedScalar scale,
                          bool observe, bool allow_mixed_basis, CeedOperator *op) {
  CeedBasis            base, basis;
  CeedElemRestriction  restriction;
  CeedQFunction        qf;
  CeedQFunctionContext context;
  const CeedScalar    *interp, *grad, *q_ref, *q_weight;
  CeedInt              nodes = 1, size = num_comp * (gradient ? dim : 1);
  for (CeedInt i = 0; i < dim; i++) nodes *= p;
  CeedCall(CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p, q, CEED_GAUSS, &base));
  CeedCall(CeedBasisGetInterp1D(base, &interp));
  CeedCall(CeedBasisGetGrad1D(base, &grad));
  CeedCall(CeedBasisGetQRef(base, &q_ref));
  CeedCall(CeedBasisGetQWeights(base, &q_weight));
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
  CeedCall(basis_error);
  CeedInt *offsets = malloc(num_elem * nodes * sizeof(CeedInt));
  for (CeedInt e = 0; e < num_elem; e++) {
    for (CeedInt n = 0; n < nodes; n++) offsets[e * nodes + n] = e * (nodes - 1) + n;
  }

  if (num_elem && !offsets) return CEED_ERROR_MAJOR;
  const int restriction_error = CeedElemRestrictionCreate(ceed, num_elem, nodes, num_comp, num_elem ? nodes * num_elem : 1,
                                                          nodes * num_comp * num_elem, CEED_MEM_HOST, CEED_COPY_VALUES, offsets, &restriction);
  free(offsets);
  CeedCall(restriction_error);
  CeedCall(CeedQFunctionCreateInterior(ceed, 1, ScaleComponents, ScaleComponents_loc, &qf));
  CeedCall(CeedQFunctionAddInput(qf, "u", size, gradient ? CEED_EVAL_GRAD : CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionAddOutput(qf, "v", size, gradient ? CEED_EVAL_GRAD : CEED_EVAL_INTERP));
  const bool mixed_basis = allow_mixed_basis && dim == 3 && gradient && p < q;
  if (mixed_basis) CeedCall(CeedQFunctionAddInput(qf, "extra", 1, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionContextCreate(ceed, &context));
  PipelineContext context_data = {size, 0, observe, mixed_basis, 1};
  CeedCall(CeedQFunctionContextSetData(context, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof context_data, &context_data));
  CeedCall(CeedQFunctionSetContext(qf, context));
  CeedCall(CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op));
  CeedCall(CeedOperatorSetField(*op, "u", restriction, basis, CEED_VECTOR_ACTIVE));
  CeedCall(CeedOperatorSetField(*op, "v", restriction, basis, CEED_VECTOR_ACTIVE));
  if (mixed_basis) {
    // This larger passive Basis makes T>Q while the active gradient still uses
    // collocated slices. Collectives must include the padded x/y threads too.
    CeedBasis           extra_basis;
    CeedElemRestriction extra_restriction;
    CeedVector          extra;
    const CeedInt       extra_p = q + 2, extra_nodes = extra_p * extra_p * extra_p;
    const CeedInt       extra_strides[3] = {1, extra_nodes, extra_nodes};
    CeedCall(CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, extra_p, q, CEED_GAUSS, &extra_basis));
    CeedCall(CeedElemRestrictionCreateStrided(ceed, num_elem, extra_nodes, 1, extra_nodes * num_elem, extra_strides, &extra_restriction));
    CeedCall(CeedVectorCreate(ceed, extra_nodes * num_elem, &extra));
    CeedCall(CeedVectorSetValue(extra, 0));
    CeedCall(CeedOperatorSetField(*op, "extra", extra_restriction, extra_basis, extra));
    CeedCall(CeedBasisDestroy(&extra_basis));
    CeedCall(CeedElemRestrictionDestroy(&extra_restriction));
    CeedCall(CeedVectorDestroy(&extra));
  }
  CeedCall(CeedBasisDestroy(&base));
  CeedCall(CeedBasisDestroy(&basis));
  CeedCall(CeedElemRestrictionDestroy(&restriction));
  CeedCall(CeedQFunctionDestroy(&qf));
  CeedCall(CeedQFunctionContextDestroy(&context));
  return 0;
}

static int CompareVectors(CeedVector actual, CeedVector expected, CeedSize length, const char *phase) {
  const CeedScalar *result, *oracle;
  CeedScalar        max_error = 0, max_value = 0;
  bool              finite = true;
  CeedCall(CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &result));
  CeedCall(CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &oracle));
  for (CeedSize i = 0; i < length; i++) {
    finite    = finite && isfinite(result[i]) && isfinite(oracle[i]);
    max_error = fmax(max_error, fabs(result[i] - oracle[i]));
    max_value = fmax(max_value, fabs(oracle[i]));
  }
  CeedCall(CeedVectorRestoreArrayRead(actual, &result));
  CeedCall(CeedVectorRestoreArrayRead(expected, &oracle));
  if (!finite || max_error > 500 * CEED_EPSILON * fmax(1, max_value)) {
    printf("%s: error %g, reference scale %g, finite %d\n", phase, max_error, max_value, finite);
    return 1;
  }
  return 0;
}

static double Seconds(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return now.tv_sec + 1e-9 * now.tv_nsec;
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
  CeedCall(CeedInit(argv[1], &ceed));
  CeedCall(CeedInit("/cpu/self/ref/serial", &reference));
  CeedCall(CeedVectorCreate(ceed, length, &u));
  CeedCall(CeedVectorCreate(ceed, length, &v));
  CeedCall(CeedVectorCreate(reference, length, &u_ref));
  CeedCall(CeedVectorCreate(reference, length, &v_ref));
  CeedVector inputs[2] = {u, u_ref};
  for (int side = 0; side < 2; side++) {
    CeedScalar *input;
    CeedCall(CeedVectorGetArrayWrite(inputs[side], CEED_MEM_HOST, &input));
    for (CeedSize i = 0; i < length; i++) input[i] = sin(0.017 * i) + 0.2 * cos(0.037 * i);
    CeedCall(CeedVectorRestoreArray(inputs[side], &input));
  }
  const bool timing            = argc > 8 && atoi(argv[8]) > 0;
  const bool allow_mixed_basis = argc <= 9 || atoi(argv[9]);
  const bool is_cuda           = strstr(argv[1], "/gpu/cuda/gen") != NULL;
  const bool is_hip            = strstr(argv[1], "/gpu/hip/gen") != NULL;
  const bool observe           = !timing && (is_cuda || is_hip);
  for (CeedInt i = 0; i < 2; i++) {
    CeedCall(CreateOperator(ceed, dim, p, q, num_comp, num_elem, gradient, 1 + 0.2 * i, observe, allow_mixed_basis, &operators[i]));
    CeedCall(CreateOperator(reference, dim, p, q, num_comp, num_elem, gradient, 1 + 0.2 * i, observe, allow_mixed_basis, &oracles[i]));
  }
  CeedCall(CeedOperatorCreateComposite(ceed, &composite));
  CeedCall(CeedOperatorCreateComposite(reference, &reference_composite));
  for (CeedInt i = 0; i < 2; i++) {
    CeedCall(CeedOperatorCompositeAddSub(composite, operators[i]));
    CeedCall(CeedOperatorCompositeAddSub(reference_composite, oracles[i]));
  }
  // Alternate distinct same-shape modules, then capture and replay their composite.
  for (CeedInt step = 0; step < 8; step++) {
    if (step >= 5) {
      for (int i = 0; i < 2; i++) {
        CeedOperator pair[2] = {operators[i], oracles[i]};
        for (int side = 0; side < 2; side++) {
          CeedQFunctionContext context;
          PipelineContext     *values;
          CeedCall(CeedOperatorGetContext(pair[side], &context));
          CeedCall(CeedQFunctionContextGetData(context, CEED_MEM_HOST, &values));
          values->scale = 1 + 0.05 * step;
          CeedCall(CeedQFunctionContextRestoreData(context, &values));
          CeedCall(CeedQFunctionContextDestroy(&context));
          if (allow_mixed_basis && dim == 3 && gradient && p < q) {
            CeedOperatorField field;
            CeedVector        extra;
            CeedCall(CeedOperatorGetFieldByName(pair[side], "extra", &field));
            CeedCall(CeedOperatorFieldGetVector(field, &extra));
            CeedCall(CeedVectorSetValue(extra, 0.2 * step));
            CeedCall(CeedVectorDestroy(&extra));
          }
        }
      }
    }
    CeedOperator actual            = step < 4 ? operators[step % 2] : composite;
    CeedOperator expected          = step < 4 ? oracles[step % 2] : reference_composite;
    const double first_apply_start = timing && step == 0 ? Seconds() : 0;
    CeedCall(CeedOperatorApply(actual, u, v, CEED_REQUEST_IMMEDIATE));
    if (timing && step == 0) {
      const CeedScalar *values;
      CeedCall(CeedVectorGetArrayRead(v, CEED_MEM_HOST, &values));
      printf("First_apply_seconds %.12g\n", Seconds() - first_apply_start);
      CeedCall(CeedVectorRestoreArrayRead(v, &values));
    }
    CeedCall(CeedOperatorApply(expected, u_ref, v_ref, CEED_REQUEST_IMMEDIATE));
    CeedCall(CompareVectors(v, v_ref, length, "Apply and composite replay"));
    if (observe) {
      for (int i = 0; i < 2; i++) {
        CeedQFunctionContext   actual_context, expected_context;
        const PipelineContext *actual_data, *expected_data;
        CeedCall(CeedOperatorGetContext(operators[i], &actual_context));
        CeedCall(CeedOperatorGetContext(oracles[i], &expected_context));
        CeedCall(CeedQFunctionContextGetDataRead(actual_context, CEED_MEM_HOST, &actual_data));
        CeedCall(CeedQFunctionContextGetDataRead(expected_context, CEED_MEM_HOST, &expected_data));
        const CeedInt actual_calls = actual_data->calls, expected_calls = expected_data->calls;
        CeedCall(CeedQFunctionContextRestoreDataRead(actual_context, &actual_data));
        CeedCall(CeedQFunctionContextRestoreDataRead(expected_context, &expected_data));
        CeedCall(CeedQFunctionContextDestroy(&actual_context));
        CeedCall(CeedQFunctionContextDestroy(&expected_context));
        if (actual_calls != expected_calls) {
          printf("QFunction points: %d != %d\n", actual_calls, expected_calls);
          return 1;
        }
      }
    }
  }
  // Small domains exercise CUDA/CPU assembly. HIP assembly has existing failures
  // outside this Apply pipeline port and is checked by the separate assembly tests.
  if (num_elem <= 17 && !is_hip) {
    CeedCall(CeedOperatorLinearAssembleDiagonal(operators[0], v, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedOperatorLinearAssembleDiagonal(oracles[0], v_ref, CEED_REQUEST_IMMEDIATE));
    CeedCall(CompareVectors(v, v_ref, length, "Diagonal assembly"));
  }
  if (observe && is_cuda && allow_mixed_basis && dim == 3 && gradient && p < q) {
    CeedOperatorField      field;
    CeedVector             extra;
    CeedQFunctionContext   context;
    CeedScalar            *array;
    const PipelineContext *context_values;
    CeedCall(CeedSetErrorHandler(ceed, CeedErrorStore));
    CeedCall(CeedOperatorGetFieldByName(operators[0], "extra", &field));
    CeedCall(CeedOperatorFieldGetVector(field, &extra));
    CeedCall(CeedOperatorGetContext(operators[0], &context));
    CeedOperator failure_composite;
    CeedCall(CeedOperatorCreateComposite(ceed, &failure_composite));
    CeedCall(CeedOperatorCompositeAddSub(failure_composite, operators[0]));
    CeedOperator targets[2] = {operators[0], failure_composite};
    for (int target = 0; target < 2; target++) {
      CeedCall(CeedVectorGetArray(extra, CEED_MEM_DEVICE, &array));
      const int vector_error = CeedOperatorApply(targets[target], u, v, CEED_REQUEST_IMMEDIATE);
      CeedCall(CeedVectorRestoreArray(extra, &array));
      const char *message;
      CeedCall(CeedGetErrorMessage(ceed, &message));
      if (vector_error != CEED_ERROR_BACKEND || !strstr(message, "CeedVector read-only array access")) {
        printf("Expected passive borrow rejection, got %d\n", vector_error);
        return 1;
      }
      CeedCall(CeedQFunctionContextGetDataRead(context, CEED_MEM_HOST, &context_values));
      const int context_error = CeedOperatorApply(targets[target], u, v, CEED_REQUEST_IMMEDIATE);
      CeedCall(CeedQFunctionContextRestoreDataRead(context, &context_values));
      CeedCall(CeedGetErrorMessage(ceed, &message));
      if (context_error != CEED_ERROR_BACKEND || !strstr(message, "CeedQFunctionContext data access")) {
        printf("Expected context borrow rejection, got %d\n", context_error);
        return 1;
      }
      // This writable borrow rejects any reader leaked by the context failure.
      CeedCall(CeedVectorGetArray(extra, CEED_MEM_DEVICE, &array));
      CeedCall(CeedVectorRestoreArray(extra, &array));
    }
    CeedCall(CeedOperatorDestroy(&failure_composite));
    CeedCall(CeedOperatorApply(operators[0], u, v, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedOperatorApply(oracles[0], u_ref, v_ref, CEED_REQUEST_IMMEDIATE));
    CeedCall(CompareVectors(v, v_ref, length, "Retry after partial binding rejection"));
    CeedCall(CeedQFunctionContextDestroy(&context));
    CeedCall(CeedVectorDestroy(&extra));
  }
  if (timing) {
    const int         repeats = atoi(argv[8]);
    const CeedScalar *values;
    const double      start = Seconds();
    for (int i = 0; i < repeats; i++) CeedCall(CeedOperatorApply(operators[0], u, v, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedVectorGetArrayRead(v, CEED_MEM_HOST, &values));  // Complete GPU execution before stopping the clock.
    const double elapsed = Seconds() - start;
    CeedCall(CeedVectorRestoreArrayRead(v, &values));
    printf("Apply_seconds %.12g\n", elapsed / repeats);
  }
  // Retire graph/module owners immediately after asynchronous submission. Vectors
  // outlive the operator, so the result is independently observable afterwards.
  CeedCall(CeedOperatorApply(composite, u, v, CEED_REQUEST_IMMEDIATE));
  CeedCall(CeedOperatorApply(reference_composite, u_ref, v_ref, CEED_REQUEST_IMMEDIATE));
  CeedCall(CeedOperatorDestroy(&composite));
  CeedCall(CeedOperatorDestroy(&reference_composite));
  for (CeedInt i = 0; i < 2; i++) {
    CeedCall(CeedOperatorDestroy(&operators[i]));
    CeedCall(CeedOperatorDestroy(&oracles[i]));
  }
  CeedCall(CompareVectors(v, v_ref, length, "After operator destruction"));
  if (observe && dim == 3 && p == 3 && q == 4 && num_comp == 1 && num_elem == 1) {
    // Rename a kernel or constant symbol through the public JiT define API.
    // Driver lookup must fail without publishing a half-initialized operator.
    const char *definitions[] = {
        is_cuda ? "CeedKernelCudaGenOperator_ScaleComponents=CeedKernelCudaGenOperator_Unresolved"
                : "CeedKernelHipGenOperator_ScaleComponents=CeedKernelHipGenOperator_Unresolved",
        "ceed_basis_coefficients=ceed_basis_coefficients_unresolved"
    };
    for (int boundary = 0; boundary < 2; boundary++) {
      Ceed         failing;
      CeedOperator failure;
      CeedVector   failure_input, failure_output;
      CeedCall(CeedInit(argv[1], &failing));
      CeedCall(CeedSetErrorHandler(failing, CeedErrorStore));
      CeedCall(CeedAddJitDefine(failing, definitions[boundary]));
      CeedCall(CreateOperator(failing, dim, p, q, num_comp, num_elem, true, 1, false, false, &failure));
      CeedCall(CeedVectorCreate(failing, length, &failure_input));
      CeedCall(CeedVectorCreate(failing, length, &failure_output));
      for (int attempt = 0; attempt < 2; attempt++) {
        const char *message;
        CeedCall(CeedVectorSetValue(failure_input, 1));
        const int error = CeedOperatorApply(failure, failure_input, failure_output, CEED_REQUEST_IMMEDIATE);
        CeedCall(CeedGetErrorMessage(failing, &message));
        if (!error || !strstr(message, is_cuda ? "CUDA_ERROR_NOT_FOUND" : "hipErrorNotFound")) {
          printf("Expected kernel lookup rejection, got %d: %s\n", error, message);
          return 1;
        }
        // Both loans must have retired under the returning error handler.
        CeedCall(CeedVectorSetValue(failure_input, 0));
        CeedCall(CeedVectorSetValue(failure_output, 0));
      }
      CeedCall(CeedOperatorDestroy(&failure));
      CeedCall(CeedVectorDestroy(&failure_input));
      CeedCall(CeedVectorDestroy(&failure_output));
      CeedCall(CeedDestroy(&failing));
    }
  }
  CeedCall(CeedVectorDestroy(&u));
  CeedCall(CeedVectorDestroy(&v));
  CeedCall(CeedVectorDestroy(&u_ref));
  CeedCall(CeedVectorDestroy(&v_ref));
  CeedCall(CeedDestroy(&ceed));
  CeedCall(CeedDestroy(&reference));
  return 0;
}
