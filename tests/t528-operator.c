/// @file
/// Check quadrature coefficient register carry with more inputs than outputs.
//TESTARGS(name="backend strides") {ceed_resource} 3 4 1 7 0
//TESTARGS(name="component interleaved") {ceed_resource} 3 4 2 7 1
//TESTARGS(name="padded strides") {ceed_resource} 5 6 2 17 2
//TESTARGS(name="standard coefficients") {ceed_resource} 3 4 2 7 3
//TESTARGS(name="oriented coefficients") {ceed_resource} 3 4 2 7 4
//TESTARGS(name="curl-oriented coefficients") {ceed_resource} 3 4 2 7 5
//TESTARGS(name="direct underintegrated gradient") {ceed_resource} 6 4 2 7 1
//TESTARGS(name="single element") {ceed_resource} 3 4 1 1 2
//TESTARGS(name="empty") {ceed_resource} 3 4 1 0 0
#include "t528-operator.h"
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static int Create(Ceed ceed, CeedInt p, CeedInt q, CeedInt components, CeedInt elements, int layout, CeedOperator *op, CeedVector *input,
                  CeedVector *output, CeedVector *report, CeedVector coefficients[3]) {
  const CeedInt        nodes = p * p * p, points = q * q * q;
  CeedBasis            basis;
  CeedElemRestriction  restriction, report_restriction;
  CeedQFunction        qf;
  CeedQFunctionContext context;

  CeedBasisCreateTensorH1Lagrange(ceed, 3, components, p, q, CEED_GAUSS, &basis);
  CeedInt offsets[(elements ? elements : 1) * nodes];

  for (CeedInt e = 0; e < elements; e++) {
    for (CeedInt n = 0; n < nodes; n++) offsets[e * nodes + n] = e * (nodes - 1) + n;
  }
  CeedElemRestrictionCreate(ceed, elements, nodes, components, elements * nodes, elements * nodes * components, CEED_MEM_HOST, CEED_COPY_VALUES,
                            offsets, &restriction);
  CeedVectorCreate(ceed, elements * nodes * components, input);
  CeedVectorCreate(ceed, elements * nodes * components, output);
  CeedScalar *input_arr;

  CeedVectorGetArrayWrite(*input, CEED_MEM_HOST, &input_arr);
  for (CeedInt i = 0; i < elements * nodes * components; i++) input_arr[i] = sin(0.13 * i) + 0.2 * cos(0.07 * i);
  CeedVectorRestoreArray(*input, &input_arr);
  CeedQFunctionCreateInterior(ceed, 1, SliceCoefficients, SliceCoefficients_loc, &qf);
  CeedQFunctionAddInput(qf, "du", 3 * components, CEED_EVAL_GRAD);
  const char   *names[3] = {"scale", "diagonal", "bias"};
  const CeedInt sizes[3] = {1, 3, 1};

  for (CeedInt field = 0; field < 3; field++) CeedQFunctionAddInput(qf, names[field], sizes[field], CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf, "dv", 3 * components, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf, "report", 1, CEED_EVAL_NONE);
  CeedQFunctionContextCreate(ceed, &context);
  CeedQFunctionContextSetData(context, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof components, &components);
  CeedQFunctionSetContext(qf, context);
  CeedQFunctionSetContextWritable(qf, false);
  CeedOperatorCreate(ceed, qf, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op);
  CeedOperatorSetField(*op, "du", restriction, basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*op, "dv", restriction, basis, CEED_VECTOR_ACTIVE);
  for (CeedInt field = 0; field < 3; field++) {
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

    if (layout >= 3) {
      CeedInt indices[(elements ? elements : 1) * points];

      for (CeedInt i = 0; i < elements * points; i++) indices[i] = i;
      if (layout == 4) {
        bool orients[(elements ? elements : 1) * points];

        for (CeedInt i = 0; i < elements * points; i++) orients[i] = i % 2;
        CeedElemRestrictionCreateOriented(ceed, elements, points, count, elements * points, length, CEED_MEM_HOST, CEED_COPY_VALUES, indices, orients,
                                          &coeff_restriction);
      } else if (layout == 5) {
        CeedInt8 orients[3 * (elements ? elements : 1) * points];

        for (CeedInt i = 0; i < elements * points; i++) {
          orients[3 * i]     = i % points ? 1 : 0;
          orients[3 * i + 1] = i % 2 ? -1 : 1;
          orients[3 * i + 2] = i % points + 1 < points ? -1 : 0;
        }
        CeedElemRestrictionCreateCurlOriented(ceed, elements, points, count, elements * points, length, CEED_MEM_HOST, CEED_COPY_VALUES, indices,
                                              orients, &coeff_restriction);
      } else {
        CeedElemRestrictionCreate(ceed, elements, points, count, elements * points, length, CEED_MEM_HOST, CEED_COPY_VALUES, indices,
                                  &coeff_restriction);
      }
    } else {
      CeedElemRestrictionCreateStrided(ceed, elements, points, count, length, layout == 0 ? CEED_STRIDES_BACKEND : strides, &coeff_restriction);
    }
    if (layout == 0) CeedElemRestrictionGetLLayout(coeff_restriction, strides);
    CeedVectorCreate(ceed, length, &coefficients[field]);
    CeedScalar *coefficients_arr;

    CeedVectorSetValue(coefficients[field], 0);
    CeedVectorGetArray(coefficients[field], CEED_MEM_HOST, &coefficients_arr);
    for (CeedInt e = 0; e < elements; e++) {
      for (CeedInt n = 0; n < points; n++) {
        for (CeedInt c = 0; c < count; c++)
          coefficients_arr[e * strides[2] + n * strides[0] + c * strides[1]] = 0.2 + 0.017 * n + 0.07 * e + 0.03 * c + 0.11 * field;
      }
    }
    CeedVectorRestoreArray(coefficients[field], &coefficients_arr);
    CeedOperatorSetField(*op, names[field], coeff_restriction, CEED_BASIS_NONE, coefficients[field]);
    CeedElemRestrictionDestroy(&coeff_restriction);
  }
  const CeedInt report_strides[3] = {1, elements * points, points};

  CeedElemRestrictionCreateStrided(ceed, elements, points, 1, elements * points, report_strides, &report_restriction);
  CeedVectorCreate(ceed, elements * points, report);
  CeedOperatorSetField(*op, "report", report_restriction, CEED_BASIS_NONE, *report);
  CeedElemRestrictionDestroy(&report_restriction);
  CeedElemRestrictionDestroy(&restriction);
  CeedBasisDestroy(&basis);
  CeedQFunctionContextDestroy(&context);
  CeedQFunctionDestroy(&qf);
  return CEED_ERROR_SUCCESS;
}

int main(int argc, char **argv) {
  const CeedInt p = atoi(argv[2]), q = atoi(argv[3]), components = atoi(argv[4]), elements = atoi(argv[5]);
  const int     layout = atoi(argv[6]);
  Ceed          ceed, ceed_ref;
  CeedOperator  op, op_ref, op_composite, op_composite_ref;
  CeedVector    u, u_ref, v, v_ref, report, report_ref, coefficients[3], coefficients_ref[3];

  CeedInit(argv[1], &ceed);
  CeedInit("/cpu/self/ref/serial", &ceed_ref);
  Create(ceed, p, q, components, elements, layout, &op, &u, &v, &report, coefficients);
  Create(ceed_ref, p, q, components, elements, layout, &op_ref, &u_ref, &v_ref, &report_ref, coefficients_ref);
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCreateComposite(ceed_ref, &op_composite_ref);
  CeedOperatorCompositeAddSub(op_composite, op);
  CeedOperatorCompositeAddSub(op_composite_ref, op_ref);
  if (layout == 4 || layout == 5) {
    // Assembly before Apply must also honor unsupported restriction types
    CeedVector          assembled[2];
    CeedElemRestriction restriction[2];
    CeedSize            length;

    for (CeedInt side = 0; side < 2; side++)
      CeedOperatorLinearAssembleQFunction(side ? op_ref : op, &assembled[side], &restriction[side], CEED_REQUEST_IMMEDIATE);
    CeedVectorGetLength(assembled[0], &length);
    CeedVector        ordered;
    const CeedScalar *assembled_arr;
    CeedScalar       *ordered_arr;
    CeedInt           strides[2][3], count;

    // QFunction assembly layouts can differ between backends
    for (CeedInt side = 0; side < 2; side++) CeedElemRestrictionGetLLayout(restriction[side], strides[side]);
    CeedElemRestrictionGetNumComponents(restriction[0], &count);
    CeedVectorCreate(ceed_ref, length, &ordered);
    CeedVectorGetArrayRead(assembled[0], CEED_MEM_HOST, &assembled_arr);
    CeedVectorGetArrayWrite(ordered, CEED_MEM_HOST, &ordered_arr);
    for (CeedInt e = 0; e < elements; e++) {
      for (CeedInt qpt = 0; qpt < q * q * q; qpt++) {
        for (CeedInt c = 0; c < count; c++)
          ordered_arr[qpt * strides[1][0] + c * strides[1][1] + e * strides[1][2]] =
              assembled_arr[qpt * strides[0][0] + c * strides[0][1] + e * strides[0][2]];
      }
    }
    CeedVectorRestoreArrayRead(assembled[0], &assembled_arr);
    CeedVectorRestoreArray(ordered, &ordered_arr);
    CompareVectors(ordered, assembled[1], "QFunction assembly");
    CeedVectorDestroy(&ordered);
    for (CeedInt side = 0; side < 2; side++) {
      CeedVectorDestroy(&assembled[side]);
      CeedElemRestrictionDestroy(&restriction[side]);
    }
  }
  for (CeedInt step = 0; step < 5; step++) {
    if (step == 3) {
      CeedVectorSetValue(coefficients[2], 0.7);
      CeedVectorSetValue(coefficients_ref[2], 0.7);
    }
    CeedOperatorApply(step < 2 ? op : op_composite, u, v, CEED_REQUEST_IMMEDIATE);
    CeedOperatorApply(step < 2 ? op_ref : op_composite_ref, u_ref, v_ref, CEED_REQUEST_IMMEDIATE);
    CompareVectors(v, v_ref, "Coefficient carry");
    CompareVectors(report, report_ref, "Point output");
  }
  CeedOperatorDestroy(&op_composite);
  CeedOperatorDestroy(&op_composite_ref);
  CeedOperatorDestroy(&op);
  CeedOperatorDestroy(&op_ref);
  for (CeedInt field = 0; field < 3; field++) {
    CeedVectorDestroy(&coefficients[field]);
    CeedVectorDestroy(&coefficients_ref[field]);
  }
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&u_ref);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&v_ref);
  CeedVectorDestroy(&report);
  CeedVectorDestroy(&report_ref);
  CeedDestroy(&ceed);
  CeedDestroy(&ceed_ref);
  return CEED_ERROR_SUCCESS;
}
