/// @file
/// Test setting QFunctionContext fields from Operator
/// \test Test setting QFunctionContext fields from Operator
#include <ceed.h>
#include <stddef.h>
#include <stdio.h>

#include "t500-operator.h"

typedef struct {
  int    count;
  double other;
} TestContext1;

typedef struct {
  double time;
  double other;
} TestContext2;

int main(int argc, char **argv) {
  Ceed                  ceed;
  CeedQFunctionContext  qf_ctx_sub_1, qf_ctx_sub_2;
  CeedContextFieldLabel count_label, other_label, time_label, bad_label;
  CeedQFunction         qf_sub_1, qf_sub_2;
  CeedOperator          op_sub_1, op_sub_2, op_composite;

  TestContext1 ctx_data_1 = {
      .count = 42,
      .other = -3.0,
  };
  TestContext2 ctx_data_2 = {
      .time  = 1.0,
      .other = -3.0,
  };

  CeedInit(argv[1], &ceed);

  // First sub-operator
  CeedQFunctionContextCreate(ceed, &qf_ctx_sub_1);
  CeedQFunctionContextSetData(qf_ctx_sub_1, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(TestContext1), &ctx_data_1);
  CeedQFunctionContextRegisterInt32(qf_ctx_sub_1, "count", offsetof(TestContext1, count), 1, "some sort of counter");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_1, "other", offsetof(TestContext1, other), 1, "some other value");

  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_sub_1);
  CeedQFunctionSetContext(qf_sub_1, qf_ctx_sub_1);

  CeedOperatorCreate(ceed, qf_sub_1, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_sub_1);

  // Check setting field in operator
  CeedOperatorGetContextFieldLabel(op_sub_1, "count", &count_label);
  int value_count = 43;
  CeedOperatorSetContextInt32(op_sub_1, count_label, &value_count);
  if (ctx_data_1.count != 43) printf("Incorrect context data for count: %" CeedInt_FMT " != 43", ctx_data_1.count);
  {
    const int *values;
    size_t     num_values;

    CeedOperatorGetContextInt32Read(op_sub_1, count_label, &num_values, &values);
    if (num_values != 1) printf("Incorrect number of count values, found %ld but expected 1", num_values);
    if (values[0] != ctx_data_1.count) printf("Incorrect value found, found %d but expected %d", values[0], ctx_data_1.count);
    CeedOperatorRestoreContextInt32Read(op_sub_1, count_label, &values);
  }

  // Second sub-operator
  CeedQFunctionContextCreate(ceed, &qf_ctx_sub_2);
  CeedQFunctionContextSetData(qf_ctx_sub_2, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(TestContext2), &ctx_data_2);
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_2, "time", offsetof(TestContext2, time), 1, "current time");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_2, "other", offsetof(TestContext2, other), 1, "some other value");

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_sub_2);
  CeedQFunctionSetContext(qf_sub_2, qf_ctx_sub_2);

  CeedOperatorCreate(ceed, qf_sub_2, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_sub_2);

  // Composite operator
  CeedCompositeOperatorCreate(ceed, &op_composite);
  CeedCompositeOperatorAddSub(op_composite, op_sub_1);
  CeedCompositeOperatorAddSub(op_composite, op_sub_2);

  // Check setting field in context of single sub-operator for composite operator
  CeedOperatorGetContextFieldLabel(op_composite, "time", &time_label);
  double value_time = 2.0;
  CeedOperatorSetContextDouble(op_composite, time_label, &value_time);
  if (ctx_data_2.time != 2.0) printf("Incorrect context data for time: %f != 2.0\n", ctx_data_2.time);
  {
    const double *values;
    size_t        num_values;

    CeedOperatorGetContextDoubleRead(op_composite, time_label, &num_values, &values);
    if (num_values != 1) printf("Incorrect number of time values, found %ld but expected 1", num_values);
    if (values[0] != ctx_data_2.time) printf("Incorrect value found, found %f but expected %f", values[0], ctx_data_2.time);
    CeedOperatorRestoreContextDoubleRead(op_composite, time_label, &values);
  }

  // Check setting field in context of multiple sub-operators for composite operator
  CeedOperatorGetContextFieldLabel(op_composite, "other", &other_label);
  // No issue requesting same label twice
  CeedOperatorGetContextFieldLabel(op_composite, "other", &other_label);
  double value_other = 9000.;
  CeedOperatorSetContextDouble(op_composite, other_label, &value_other);
  if (ctx_data_1.other != 9000.0) printf("Incorrect context data for other: %f != 2.0\n", ctx_data_1.other);
  if (ctx_data_2.other != 9000.0) printf("Incorrect context data for other: %f != 2.0\n", ctx_data_2.other);

  // Check requesting label for field that doesn't exist returns NULL
  CeedOperatorGetContextFieldLabel(op_composite, "bad", &bad_label);
  if (bad_label) printf("Incorrect context label returned\n");

  {
    // Check getting reference to QFunctionContext
    CeedQFunctionContext ctx_copy = NULL;

    CeedOperatorGetContext(op_sub_1, &ctx_copy);
    if (ctx_copy != qf_ctx_sub_1) printf("Incorrect QFunctionContext retrieved");

    CeedOperatorGetContext(op_sub_2, &ctx_copy);  // Destroys reference to qf_ctx_sub_1
    if (ctx_copy != qf_ctx_sub_2) printf("Incorrect QFunctionContext retrieved");
    CeedQFunctionContextDestroy(&ctx_copy);  // Cleanup to prevent leak
  }

  CeedQFunctionContextDestroy(&qf_ctx_sub_1);
  CeedQFunctionContextDestroy(&qf_ctx_sub_2);
  CeedQFunctionDestroy(&qf_sub_1);
  CeedQFunctionDestroy(&qf_sub_2);
  CeedOperatorDestroy(&op_sub_1);
  CeedOperatorDestroy(&op_sub_2);
  CeedOperatorDestroy(&op_composite);
  CeedDestroy(&ceed);
  return 0;
}
