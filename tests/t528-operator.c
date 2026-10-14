/// @file
/// Check quadrature coefficient register carry with more inputs than outputs.
//TESTARGS(name="backend strides") {ceed_resource} 3 4 1 7 0
//TESTARGS(name="component interleaved") {ceed_resource} 3 4 2 7 1
//TESTARGS(name="padded strides") {ceed_resource} 5 6 2 17 2
//TESTARGS(name="standard coefficients") {ceed_resource} 3 4 2 7 3
//TESTARGS(name="direct underintegrated gradient") {ceed_resource} 6 4 2 7 1
//TESTARGS(name="single element") {ceed_resource} 3 4 1 1 2
//TESTARGS(name="empty") {ceed_resource} 3 4 1 0 0
#include "t528-operator.h"
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int Compare(CeedVector actual, CeedVector expected, CeedSize length) {
  const CeedScalar *a, *b;
  CeedScalar        error = 0, scale = 0;
  bool              finite = true;
  CeedCall(CeedVectorGetArrayRead(actual, CEED_MEM_HOST, &a));
  CeedCall(CeedVectorGetArrayRead(expected, CEED_MEM_HOST, &b));
  for (CeedSize i = 0; i < length; i++) {
    finite = finite && isfinite(a[i]) && isfinite(b[i]);
    error  = fmax(error, fabs(a[i] - b[i]));
    scale  = fmax(scale, fabs(b[i]));
  }
  CeedCall(CeedVectorRestoreArrayRead(actual, &a));
  CeedCall(CeedVectorRestoreArrayRead(expected, &b));
  if (!finite || error > 500 * CEED_EPSILON * fmax(1, scale)) {
    printf("Coefficient carry: error %g, scale %g, finite %d\n", error, scale, finite);
    return 1;
  }
  return 0;
}

static int Create(Ceed ceed, CeedInt p, CeedInt q, CeedInt components, CeedInt elements, int layout, CeedOperator *op, CeedVector *input,
                  CeedVector *output, CeedVector *report, CeedVector coefficients[3]) {
  const CeedInt        nodes = p * p * p, points = q * q * q;
  CeedBasis            basis;
  CeedElemRestriction  restriction, report_restriction;
  CeedQFunction        qf;
  CeedQFunctionContext context;
  CeedCall(CeedBasisCreateTensorH1Lagrange(ceed, 3, components, p, q, CEED_GAUSS, &basis));
  CeedInt *offsets = malloc(elements * nodes * sizeof *offsets);
  for (CeedInt e = 0; e < elements; e++) {
    for (CeedInt n = 0; n < nodes; n++) offsets[e * nodes + n] = e * (nodes - 1) + n;
  }
  if (elements && !offsets) return CEED_ERROR_MAJOR;
  const int restriction_error = CeedElemRestrictionCreate(ceed, elements, nodes, components, elements * nodes, elements * nodes * components,
                                                          CEED_MEM_HOST, CEED_COPY_VALUES, offsets, &restriction);
  free(offsets);
  CeedCall(restriction_error);
  CeedCall(CeedVectorCreate(ceed, elements * nodes * components, input));
  CeedCall(CeedVectorCreate(ceed, elements * nodes * components, output));
  CeedScalar *values;
  CeedCall(CeedVectorGetArrayWrite(*input, CEED_MEM_HOST, &values));
  for (CeedInt i = 0; i < elements * nodes * components; i++) values[i] = sin(0.13 * i) + 0.2 * cos(0.07 * i);
  CeedCall(CeedVectorRestoreArray(*input, &values));
  CeedCall(CeedQFunctionCreateInterior(ceed, 1, SliceCoefficients, SliceCoefficients_loc, &qf));
  CeedCall(CeedQFunctionAddInput(qf, "du", 3 * components, CEED_EVAL_GRAD));
  const char   *names[3] = {"scale", "diagonal", "bias"};
  const CeedInt sizes[3] = {1, 3, 1};
  for (int field = 0; field < 3; field++) CeedCall(CeedQFunctionAddInput(qf, names[field], sizes[field], CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf, "dv", 3 * components, CEED_EVAL_GRAD));
  CeedCall(CeedQFunctionAddOutput(qf, "report", 1, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionContextCreate(ceed, &context));
  CeedCall(CeedQFunctionContextSetData(context, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof components, &components));
  CeedCall(CeedQFunctionSetContext(qf, context));
  CeedCall(CeedQFunctionSetContextWritable(qf, false));
  CeedCall(CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op));
  CeedCall(CeedOperatorSetField(*op, "du", restriction, basis, CEED_VECTOR_ACTIVE));
  CeedCall(CeedOperatorSetField(*op, "dv", restriction, basis, CEED_VECTOR_ACTIVE));
  for (int field = 0; field < 3; field++) {
    const CeedInt count      = sizes[field];
    CeedInt       strides[3] = {1, elements * points, points};
    if (layout == 1) {
      strides[0] = count;
      strides[1] = 1;
      strides[2] = count * points;
    }
    if (layout == 2) {
      strides[0] = 2;
      strides[1] = 2 * points + 3;
      strides[2] = count * (2 * points + 3) + 5;
    }
    const CeedSize      length = elements ? (CeedSize)(elements - 1) * strides[2] + (points - 1) * strides[0] + (count - 1) * strides[1] + 1 : 0;
    CeedElemRestriction coeff_restriction;
    if (layout == 3) {
      CeedInt *indices = malloc(elements * points * sizeof *indices);
      for (CeedInt i = 0; i < elements * points; i++) indices[i] = i;
      if (elements && !indices) return CEED_ERROR_MAJOR;
      const int error = CeedElemRestrictionCreate(ceed, elements, points, count, elements * points, length, CEED_MEM_HOST, CEED_COPY_VALUES, indices,
                                                  &coeff_restriction);
      free(indices);
      CeedCall(error);
    } else {
      CeedCall(CeedElemRestrictionCreateStrided(ceed, elements, points, count, length, layout == 0 ? CEED_STRIDES_BACKEND : strides,
                                                &coeff_restriction));
    }
    if (layout == 0) CeedCall(CeedElemRestrictionGetLLayout(coeff_restriction, strides));
    CeedCall(CeedVectorCreate(ceed, length, &coefficients[field]));
    CeedScalar *coeffs;
    CeedCall(CeedVectorSetValue(coefficients[field], 0));
    CeedCall(CeedVectorGetArray(coefficients[field], CEED_MEM_HOST, &coeffs));
    for (CeedInt e = 0; e < elements; e++) {
      for (CeedInt n = 0; n < points; n++) {
        for (CeedInt c = 0; c < count; c++)
          coeffs[e * strides[2] + n * strides[0] + c * strides[1]] = 0.2 + 0.017 * n + 0.07 * e + 0.03 * c + 0.11 * field;
      }
    }
    CeedCall(CeedVectorRestoreArray(coefficients[field], &coeffs));
    CeedCall(CeedOperatorSetField(*op, names[field], coeff_restriction, CEED_BASIS_NONE, coefficients[field]));
    CeedCall(CeedElemRestrictionDestroy(&coeff_restriction));
  }
  const CeedInt report_strides[3] = {1, elements * points, points};
  CeedCall(CeedElemRestrictionCreateStrided(ceed, elements, points, 1, elements * points, report_strides, &report_restriction));
  CeedCall(CeedVectorCreate(ceed, elements * points, report));
  CeedCall(CeedOperatorSetField(*op, "report", report_restriction, CEED_BASIS_NONE, *report));
  CeedCall(CeedElemRestrictionDestroy(&report_restriction));
  CeedCall(CeedElemRestrictionDestroy(&restriction));
  CeedCall(CeedBasisDestroy(&basis));
  CeedCall(CeedQFunctionContextDestroy(&context));
  CeedCall(CeedQFunctionDestroy(&qf));
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 7) return 1;
  const CeedInt p = atoi(argv[2]), q = atoi(argv[3]), components = atoi(argv[4]), elements = atoi(argv[5]);
  const int     layout = atoi(argv[6]);
  Ceed          ceed[2];
  CeedOperator  operators[2], composites[2];
  CeedVector    input[2], output[2], report[2], coefficients[2][3];
  CeedCall(CeedInit(argv[1], &ceed[0]));
  CeedCall(CeedInit("/cpu/self/ref/serial", &ceed[1]));
  for (int side = 0; side < 2; side++) {
    CeedCall(Create(ceed[side], p, q, components, elements, layout, &operators[side], &input[side], &output[side], &report[side],
                    coefficients[side]));
    CeedCall(CeedOperatorCreateComposite(ceed[side], &composites[side]));
    CeedCall(CeedOperatorCompositeAddSub(composites[side], operators[side]));
  }
  for (int step = 0; step < 5; step++) {
    if (step == 3) {
      for (int side = 0; side < 2; side++) CeedCall(CeedVectorSetValue(coefficients[side][2], 0.7));
    }
    for (int side = 0; side < 2; side++)
      CeedCall(CeedOperatorApply(step < 2 ? operators[side] : composites[side], input[side], output[side], CEED_REQUEST_IMMEDIATE));
    CeedCall(Compare(output[0], output[1], elements * p * p * p * components));
    CeedCall(Compare(report[0], report[1], elements * q * q * q));
  }
  for (int side = 0; side < 2; side++) {
    CeedCall(CeedOperatorDestroy(&composites[side]));
    CeedCall(CeedOperatorDestroy(&operators[side]));
    for (int field = 0; field < 3; field++) CeedCall(CeedVectorDestroy(&coefficients[side][field]));
    CeedCall(CeedVectorDestroy(&input[side]));
    CeedCall(CeedVectorDestroy(&output[side]));
    CeedCall(CeedVectorDestroy(&report[side]));
    CeedCall(CeedDestroy(&ceed[side]));
  }
  return 0;
}
