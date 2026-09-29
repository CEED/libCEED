/// @file
/// Test setting QFunctionContext fields from Operator
/// \test Test setting QFunctionContext fields from Operator
#include <ceed.h>
#include <stddef.h>
#include <stdio.h>

#include "t500-operator.h"

typedef struct {
  bool       value_bool;
  char       value_byte;
  CeedInt8   value_int8;
  CeedInt    value_int;
  int32_t    value_int32;
  int64_t    value_int64;
  CeedSize   value_size;
  CeedScalar value_scalar;
  float      value_float;
  double     value_double;
  int32_t    count;
  double     other;
  bool       type_mismatch;
  double     num_values_mismatch[2];
} TestContext1;

typedef struct {
  bool       value_bool;
  char       value_byte;
  CeedInt8   value_int8;
  CeedInt    value_int;
  int32_t    value_int32;
  int64_t    value_int64;
  CeedSize   value_size;
  CeedScalar value_scalar;
  float      value_float;
  double     value_double;
  double     time;
  double     other;
  double     type_mismatch;
  double     num_values_mismatch[3];
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
  CeedQFunctionContextRegisterBoolean(qf_ctx_sub_1, "bool", offsetof(TestContext1, value_bool), 1, "boolean value");
  CeedQFunctionContextRegisterByte(qf_ctx_sub_1, "byte", offsetof(TestContext1, value_byte), 1, "byte value");
  CeedQFunctionContextRegisterCeedInt8(qf_ctx_sub_1, "int8", offsetof(TestContext1, value_int8), 1, "8 bit integer value");
  CeedQFunctionContextRegisterCeedInt(qf_ctx_sub_1, "int", offsetof(TestContext1, value_int), 1, "CeedInt value");
  CeedQFunctionContextRegisterInt32(qf_ctx_sub_1, "int32", offsetof(TestContext1, value_int32), 1, "some sort of int32er");
  CeedQFunctionContextRegisterInt64(qf_ctx_sub_1, "int64", offsetof(TestContext1, value_int64), 1, "64 bit integer value");
  CeedQFunctionContextRegisterCeedSize(qf_ctx_sub_1, "size", offsetof(TestContext1, value_size), 1, "CeedSize value");
  CeedQFunctionContextRegisterCeedScalar(qf_ctx_sub_1, "scalar", offsetof(TestContext1, value_scalar), 1, "CeedScalar value");
  CeedQFunctionContextRegisterFloat(qf_ctx_sub_1, "float", offsetof(TestContext1, value_float), 1, "float value");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_1, "double", offsetof(TestContext1, value_double), 1, "double value");
  CeedQFunctionContextRegisterInt32(qf_ctx_sub_1, "count", offsetof(TestContext1, count), 1, "count value");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_1, "other", offsetof(TestContext1, other), 1, "other value");
  CeedQFunctionContextRegisterBoolean(qf_ctx_sub_1, "type mismatch", offsetof(TestContext1, type_mismatch), 1, "bool here, double on sub 2");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_1, "num values mismatch", offsetof(TestContext1, num_values_mismatch), 2,
                                     "2 values here, 3 on sub 2");

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
    if (num_values != 1) printf("Incorrect number of count values, found %zu but expected 1", num_values);
    if (values[0] != ctx_data_1.count) printf("Incorrect value found, found %d but expected %d", values[0], ctx_data_1.count);
    CeedOperatorRestoreContextInt32Read(op_sub_1, count_label, &values);
  }

  // Second sub-operator
  CeedQFunctionContextCreate(ceed, &qf_ctx_sub_2);
  CeedQFunctionContextSetData(qf_ctx_sub_2, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(TestContext2), &ctx_data_2);
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_2, "time", offsetof(TestContext2, time), 1, "current time");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_2, "other", offsetof(TestContext2, other), 1, "some other value");
  CeedQFunctionContextRegisterBoolean(qf_ctx_sub_2, "bool", offsetof(TestContext2, value_bool), 1, "boolean value");
  CeedQFunctionContextRegisterByte(qf_ctx_sub_2, "byte", offsetof(TestContext2, value_byte), 1, "byte value");
  CeedQFunctionContextRegisterCeedInt8(qf_ctx_sub_2, "int8", offsetof(TestContext2, value_int8), 1, "8 bit integer value");
  CeedQFunctionContextRegisterCeedInt(qf_ctx_sub_2, "int", offsetof(TestContext2, value_int), 1, "CeedInt value");
  CeedQFunctionContextRegisterInt32(qf_ctx_sub_2, "int32", offsetof(TestContext2, value_int32), 1, "some sort of int32er");
  CeedQFunctionContextRegisterInt64(qf_ctx_sub_2, "int64", offsetof(TestContext2, value_int64), 1, "64 bit integer value");
  CeedQFunctionContextRegisterCeedSize(qf_ctx_sub_2, "size", offsetof(TestContext2, value_size), 1, "CeedSize value");
  CeedQFunctionContextRegisterCeedScalar(qf_ctx_sub_2, "scalar", offsetof(TestContext2, value_scalar), 1, "CeedScalar value");
  CeedQFunctionContextRegisterFloat(qf_ctx_sub_2, "float", offsetof(TestContext2, value_float), 1, "float value");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_2, "double", offsetof(TestContext2, value_double), 1, "double value");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_2, "type mismatch", offsetof(TestContext2, type_mismatch), 1, "double here, bool on sub 1");
  CeedQFunctionContextRegisterDouble(qf_ctx_sub_2, "num values mismatch", offsetof(TestContext2, num_values_mismatch), 3,
                                     "3 values here, 2 on sub 1");
  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_sub_2);
  CeedQFunctionSetContext(qf_sub_2, qf_ctx_sub_2);

  CeedOperatorCreate(ceed, qf_sub_2, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_sub_2);

  // Composite operator
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCompositeAddSub(op_composite, op_sub_1);
  CeedOperatorCompositeAddSub(op_composite, op_sub_2);

// Check setting field in context of single sub-operator for composite operator
#define TEST_TYPE(TYPE, TYPE_CAPS, FIELD_NAME, VALUE_SET, FMT)                                                                             \
  {                                                                                                                                        \
    TYPE                  value_set = VALUE_SET;                                                                                           \
    const TYPE           *value_read;                                                                                                      \
    CeedContextFieldLabel label;                                                                                                           \
    size_t                num_values;                                                                                                      \
                                                                                                                                           \
    CeedOperatorGetContextFieldLabel(op_composite, #FIELD_NAME, &label);                                                                   \
    CeedOperatorSetContext##TYPE_CAPS(op_composite, label, &value_set);                                                                    \
    if (ctx_data_2.value_##FIELD_NAME != VALUE_SET)                                                                                        \
      printf("Incorrect context data for " #FIELD_NAME ": %" FMT " != %" FMT "\n", ctx_data_2.value_##FIELD_NAME, VALUE_SET);              \
                                                                                                                                           \
    CeedOperatorGetContext##TYPE_CAPS##Read(op_composite, label, &num_values, &value_read);                                                \
    if (num_values != 1) printf("Incorrect number of " #FIELD_NAME "values, found %zu but expected 1\n", num_values);                      \
    if (value_read[0] != VALUE_SET) printf("Incorrect value found for " #FIELD_NAME ": %" FMT " != %" FMT "\n", value_read[0], VALUE_SET); \
    CeedOperatorRestoreContext##TYPE_CAPS##Read(op_composite, label, &value_read);                                                         \
  }

  TEST_TYPE(bool, Boolean, bool, true, "d");
  TEST_TYPE(char, Byte, byte, 7, "u");
  TEST_TYPE(CeedInt8, CeedInt8, int8, -3, CeedInt8_FMT);
  TEST_TYPE(CeedInt, CeedInt, int, 74, CeedInt_FMT);
  TEST_TYPE(int32_t, Int32, int32, 111023, "d");
  TEST_TYPE(int64_t, Int64, int64, 0x1FFFFFFFF, "ld");
  TEST_TYPE(CeedScalar, CeedScalar, scalar, 1.1254e3f, "g");
  TEST_TYPE(float, Float, float, 4.22e-6f, "g");
  TEST_TYPE(double, Double, double, 5.66e21, "g");

  // Check setting field in context of single sub-operator for composite operator
  CeedOperatorGetContextFieldLabel(op_composite, "time", &time_label);
  double value_time = 2.0;
  CeedOperatorSetContextDouble(op_composite, time_label, &value_time);
  if (ctx_data_2.time != 2.0) printf("Incorrect context data for time: %f != 2.0\n", ctx_data_2.time);
  {
    const double *values;
    size_t        num_values;

    CeedOperatorGetContextDoubleRead(op_composite, time_label, &num_values, &values);
    if (num_values != 1) printf("Incorrect number of time values, found %zu but expected 1", num_values);
    if (values[0] != ctx_data_2.time) printf("Incorrect value found, found %f but expected %f\n", values[0], ctx_data_2.time);
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

  // Check requesting label for fields that don't match across sub-operators returns an error
  {
    int                   ierr;
    const char           *err_msg;
    CeedContextFieldLabel mismatch_label = NULL;

    CeedSetErrorHandler(ceed, CeedErrorStore);
    ierr = CeedOperatorGetContextFieldLabel(op_composite, "type mismatch", &mismatch_label);
    if (ierr != CEED_ERROR_INCOMPATIBLE) printf("Incompatible field types on sub-operators not detected\n");
    CeedResetErrorMessage(ceed, &err_msg);
    ierr = CeedOperatorGetContextFieldLabel(op_composite, "num values mismatch", &mismatch_label);
    if (ierr != CEED_ERROR_INCOMPATIBLE) printf("Incompatible field number of values on sub-operators not detected\n");
    CeedResetErrorMessage(ceed, &err_msg);
    CeedSetErrorHandler(ceed, CeedErrorAbort);
  }

  {
    // Check getting reference to QFunctionContext
    CeedQFunctionContext ctx_copy = NULL;

    CeedOperatorGetContext(op_sub_1, &ctx_copy);
    if (ctx_copy != qf_ctx_sub_1) printf("Incorrect QFunctionContext retrieved");
    CeedQFunctionContextDestroy(&ctx_copy);

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
