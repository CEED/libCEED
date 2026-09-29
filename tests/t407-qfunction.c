/// @file
/// Test registering and setting QFunctionContext fields
/// \test Test registering and setting QFunctionContext fields
#include <ceed.h>
#include <ceed/backend.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  bool       is_set;
  char       byte_value[2];
  CeedInt8   int8_value[2];
  CeedInt    int_value[2];
  int32_t    int32_value[2];
  int64_t    int64_value[2];
  CeedSize   size_value[2];
  CeedScalar scalar_value[2];
  float      float_value[2];
  double     time;
} TestContext;

int main(int argc, char **argv) {
  Ceed                  ceed;
  CeedQFunctionContext  ctx;
  CeedContextFieldLabel is_set_label, byte_label, int8_label, int_label;
  CeedContextFieldLabel int32_label, int64_label, size_label, scalar_label;
  CeedContextFieldLabel float_label, time_label;

  TestContext ctx_data = {
      .is_set       = true,
      .byte_value   = {1,    2   },
      .int8_value   = {2,    3   },
      .int_value    = {3,    4   },
      .int32_value  = {13,   42  },
      .int64_value  = {4,    5   },
      .size_value   = {5,    6   },
      .scalar_value = {6.0,  7.0 },
      .float_value  = {7.0f, 8.0f},
      .time         = 1.0,
  };

  CeedInit(argv[1], &ceed);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(TestContext), &ctx_data);

  CeedQFunctionContextRegisterBoolean(ctx, "is set", offsetof(TestContext, is_set), 1, "some boolean flag");
  CeedQFunctionContextRegisterByte(ctx, "byte", offsetof(TestContext, byte_value), 2, "byte values");
  CeedQFunctionContextRegisterCeedInt8(ctx, "int8", offsetof(TestContext, int8_value), 2, "8 bit integer values");
  CeedQFunctionContextRegisterCeedInt(ctx, "int", offsetof(TestContext, int_value), 2, "CeedInt values");
  CeedQFunctionContextRegisterInt32(ctx, "int32", offsetof(TestContext, int32_value), 2, "some sort of int32er");
  CeedQFunctionContextRegisterInt64(ctx, "int64", offsetof(TestContext, int64_value), 2, "64 bit integer values");
  CeedQFunctionContextRegisterCeedSize(ctx, "size", offsetof(TestContext, size_value), 2, "CeedSize values");
  CeedQFunctionContextRegisterCeedScalar(ctx, "scalar", offsetof(TestContext, scalar_value), 2, "CeedScalar values");
  CeedQFunctionContextRegisterFloat(ctx, "float", offsetof(TestContext, float_value), 2, "float values");
  CeedQFunctionContextRegisterDouble(ctx, "time", offsetof(TestContext, time), 1, "current time");

  const CeedContextFieldLabel *field_labels;
  CeedInt                      num_fields;
  CeedQFunctionContextGetAllFieldLabels(ctx, &field_labels, &num_fields);
  if (num_fields != 10) printf("Incorrect number of fields set: %" CeedInt_FMT " != 10\n", num_fields);

  const char          *name;
  size_t               num_values;
  CeedContextFieldType type;

  CeedContextFieldLabelGetDescription(field_labels[0], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "is set")) printf("Incorrect context field description for bool: \"%s\" != \"is set\"\n", name);
  if (num_values != 1) printf("Incorrect context field number of values for bool: \"%zu\" != 1\n", num_values);
  if (type != CEED_CONTEXT_FIELD_BOOL) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for bool: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_BOOL]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[1], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "byte")) printf("Incorrect context field description for byte: \"%s\" != \"byte\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for byte: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_BYTE) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for byte: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_BYTE]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[2], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "int8")) printf("Incorrect context field description for int8: \"%s\" != \"int8\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for int8: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT8) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for int8: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT8]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[3], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "int")) printf("Incorrect context field description for int: \"%s\" != \"int\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for int: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for int: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[4], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "int32")) printf("Incorrect context field description for int32: \"%s\" != \"int32\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for int32: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT32) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for int32: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT32]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[5], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "int64")) printf("Incorrect context field description for int64: \"%s\" != \"int64\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for int64: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT64) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for int64: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT64]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[6], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "size")) printf("Incorrect context field description for size: \"%s\" != \"size\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for size: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_SIZE) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for size: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_SIZE]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[7], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "scalar")) printf("Incorrect context field description for scalar: \"%s\" != \"scalar\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for scalar: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_SCALAR) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for scalar: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_SCALAR]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[8], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "float")) printf("Incorrect context field description for float: \"%s\" != \"float\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for float: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_FLOAT) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for float: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_FLOAT]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[9], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "time")) printf("Incorrect context field description for time: \"%s\" != \"time\"\n", name);
  if (num_values != 1) printf("Incorrect context field number of values for time: \"%zu\" != 1\n", num_values);
  if (type != CEED_CONTEXT_FIELD_DOUBLE) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for time: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_DOUBLE]);
    // LCOV_EXCL_STOP
  }

  // Boolean field
  CeedQFunctionContextGetFieldLabel(ctx, "is set", &is_set_label);
  bool        value_is_set = false;
  const bool *test_value_is_set;

  CeedQFunctionContextSetBoolean(ctx, is_set_label, &value_is_set);
  if (ctx_data.is_set != false) printf("Incorrect context data for is_set: %d != 0\n", ctx_data.is_set);
  CeedQFunctionContextGetBooleanRead(ctx, is_set_label, &num_values, &test_value_is_set);
  if (num_values != 1) printf("Incorrect num values for is_set: %zu != 1\n", num_values);
  if (test_value_is_set[0] != false) printf("Incorrect context data for is_set: %d != 0\n", test_value_is_set[0]);
  CeedQFunctionContextRestoreBooleanRead(ctx, is_set_label, &test_value_is_set);

  // Byte field
  CeedQFunctionContextGetFieldLabel(ctx, "byte", &byte_label);
  char        values_byte[2] = {8, 9};
  const char *test_values_byte;

  CeedQFunctionContextSetByte(ctx, byte_label, values_byte);
  if (ctx_data.byte_value[0] != 8) printf("Incorrect context data for byte[0]: %d != 8\n", ctx_data.byte_value[0]);
  if (ctx_data.byte_value[1] != 9) printf("Incorrect context data for byte[1]: %d != 9\n", ctx_data.byte_value[1]);
  CeedQFunctionContextGetByteRead(ctx, byte_label, &num_values, &test_values_byte);
  if (num_values != 2) printf("Incorrect num values for byte: %zu != 2\n", num_values);
  if (test_values_byte[0] != 8) printf("Incorrect context data for byte[0]: %d != 8\n", test_values_byte[0]);
  if (test_values_byte[1] != 9) printf("Incorrect context data for byte[1]: %d != 9\n", test_values_byte[1]);
  CeedQFunctionContextRestoreByteRead(ctx, byte_label, &test_values_byte);

  // CeedInt8 field
  CeedQFunctionContextGetFieldLabel(ctx, "int8", &int8_label);
  CeedInt8        values_int8[2] = {10, 11};
  const CeedInt8 *test_values_int8;

  CeedQFunctionContextSetCeedInt8(ctx, int8_label, values_int8);
  if (ctx_data.int8_value[0] != 10) printf("Incorrect context data for int8[0]: %" CeedInt8_FMT " != 10\n", ctx_data.int8_value[0]);
  if (ctx_data.int8_value[1] != 11) printf("Incorrect context data for int8[1]: %" CeedInt8_FMT " != 11\n", ctx_data.int8_value[1]);
  CeedQFunctionContextGetCeedInt8Read(ctx, int8_label, &num_values, &test_values_int8);
  if (num_values != 2) printf("Incorrect num values for int8: %zu != 2\n", num_values);
  if (test_values_int8[0] != 10) printf("Incorrect context data for int8[0]: %" CeedInt8_FMT " != 10\n", test_values_int8[0]);
  if (test_values_int8[1] != 11) printf("Incorrect context data for int8[1]: %" CeedInt8_FMT " != 11\n", test_values_int8[1]);
  CeedQFunctionContextRestoreCeedInt8Read(ctx, int8_label, &test_values_int8);

  // CeedInt field
  CeedQFunctionContextGetFieldLabel(ctx, "int", &int_label);
  CeedInt        values_int[2] = {12, 13};
  const CeedInt *test_values_int;

  CeedQFunctionContextSetCeedInt(ctx, int_label, values_int);
  if (ctx_data.int_value[0] != 12) printf("Incorrect context data for int[0]: %" CeedInt_FMT " != 12\n", ctx_data.int_value[0]);
  if (ctx_data.int_value[1] != 13) printf("Incorrect context data for int[1]: %" CeedInt_FMT " != 13\n", ctx_data.int_value[1]);
  CeedQFunctionContextGetCeedIntRead(ctx, int_label, &num_values, &test_values_int);
  if (num_values != 2) printf("Incorrect num values for int: %zu != 2\n", num_values);
  if (test_values_int[0] != 12) printf("Incorrect context data for int[0]: %" CeedInt_FMT " != 12\n", test_values_int[0]);
  if (test_values_int[1] != 13) printf("Incorrect context data for int[1]: %" CeedInt_FMT " != 13\n", test_values_int[1]);
  CeedQFunctionContextRestoreCeedIntRead(ctx, int_label, &test_values_int);

  // Int32 field
  CeedQFunctionContextGetFieldLabel(ctx, "int32", &int32_label);
  int32_t        values_int32[2] = {14, 43};
  const int32_t *test_values_int32;

  CeedQFunctionContextSetInt32(ctx, int32_label, values_int32);
  if (ctx_data.int32_value[0] != 14) printf("Incorrect context data for int32[0]: %" CeedInt_FMT " != 14\n", ctx_data.int32_value[0]);
  if (ctx_data.int32_value[1] != 43) printf("Incorrect context data for int32[1]: %" CeedInt_FMT " != 43\n", ctx_data.int32_value[1]);
  CeedQFunctionContextGetInt32Read(ctx, int32_label, &num_values, &test_values_int32);
  if (num_values != 2) printf("Incorrect num values for int32: %zu != 2\n", num_values);
  if (test_values_int32[0] != 14) printf("Incorrect context data for int32[0]: %" CeedInt_FMT " != 14\n", test_values_int32[0]);
  if (test_values_int32[1] != 43) printf("Incorrect context data for int32[1]: %" CeedInt_FMT " != 43\n", test_values_int32[1]);
  CeedQFunctionContextRestoreInt32Read(ctx, int32_label, &test_values_int32);

  // Int64 field
  CeedQFunctionContextGetFieldLabel(ctx, "int64", &int64_label);
  int64_t        values_int64[2] = {15, 16};
  const int64_t *test_values_int64;

  CeedQFunctionContextSetInt64(ctx, int64_label, values_int64);
  if (ctx_data.int64_value[0] != 15) printf("Incorrect context data for int64[0]: %" PRId64 " != 15\n", ctx_data.int64_value[0]);
  if (ctx_data.int64_value[1] != 16) printf("Incorrect context data for int64[1]: %" PRId64 " != 16\n", ctx_data.int64_value[1]);
  CeedQFunctionContextGetInt64Read(ctx, int64_label, &num_values, &test_values_int64);
  if (num_values != 2) printf("Incorrect num values for int64: %zu != 2\n", num_values);
  if (test_values_int64[0] != 15) printf("Incorrect context data for int64[0]: %" PRId64 " != 15\n", test_values_int64[0]);
  if (test_values_int64[1] != 16) printf("Incorrect context data for int64[1]: %" PRId64 " != 16\n", test_values_int64[1]);
  CeedQFunctionContextRestoreInt64Read(ctx, int64_label, &test_values_int64);

  // CeedSize field
  CeedQFunctionContextGetFieldLabel(ctx, "size", &size_label);
  CeedSize        values_size[2] = {17, 18};
  const CeedSize *test_values_size;

  CeedQFunctionContextSetCeedSize(ctx, size_label, values_size);
  if (ctx_data.size_value[0] != 17) printf("Incorrect context data for size[0]: %" CeedSize_FMT " != 17\n", ctx_data.size_value[0]);
  if (ctx_data.size_value[1] != 18) printf("Incorrect context data for size[1]: %" CeedSize_FMT " != 18\n", ctx_data.size_value[1]);
  CeedQFunctionContextGetCeedSizeRead(ctx, size_label, &num_values, &test_values_size);
  if (num_values != 2) printf("Incorrect num values for size: %zu != 2\n", num_values);
  if (test_values_size[0] != 17) printf("Incorrect context data for size[0]: %" CeedSize_FMT " != 17\n", test_values_size[0]);
  if (test_values_size[1] != 18) printf("Incorrect context data for size[1]: %" CeedSize_FMT " != 18\n", test_values_size[1]);
  CeedQFunctionContextRestoreCeedSizeRead(ctx, size_label, &test_values_size);

  // CeedScalar field
  CeedQFunctionContextGetFieldLabel(ctx, "scalar", &scalar_label);
  CeedScalar        values_scalar[2] = {19.0, 20.0};
  const CeedScalar *test_values_scalar;

  CeedQFunctionContextSetCeedScalar(ctx, scalar_label, values_scalar);
  if (ctx_data.scalar_value[0] != (CeedScalar)19.0) printf("Incorrect context data for scalar[0]: %g != 19.0\n", (double)ctx_data.scalar_value[0]);
  if (ctx_data.scalar_value[1] != (CeedScalar)20.0) printf("Incorrect context data for scalar[1]: %g != 20.0\n", (double)ctx_data.scalar_value[1]);
  CeedQFunctionContextGetCeedScalarRead(ctx, scalar_label, &num_values, &test_values_scalar);
  if (num_values != 2) printf("Incorrect num values for scalar: %zu != 2\n", num_values);
  if (test_values_scalar[0] != (CeedScalar)19.0) printf("Incorrect context data for scalar[0]: %g != 19.0\n", (double)test_values_scalar[0]);
  if (test_values_scalar[1] != (CeedScalar)20.0) printf("Incorrect context data for scalar[1]: %g != 20.0\n", (double)test_values_scalar[1]);
  CeedQFunctionContextRestoreCeedScalarRead(ctx, scalar_label, &test_values_scalar);

  // Float field
  CeedQFunctionContextGetFieldLabel(ctx, "float", &float_label);
  float        values_float[2] = {21.0f, 22.0f};
  const float *test_values_float;

  CeedQFunctionContextSetFloat(ctx, float_label, values_float);
  if (ctx_data.float_value[0] != 21.0f) printf("Incorrect context data for float[0]: %f != 21.0\n", ctx_data.float_value[0]);
  if (ctx_data.float_value[1] != 22.0f) printf("Incorrect context data for float[1]: %f != 22.0\n", ctx_data.float_value[1]);
  CeedQFunctionContextGetFloatRead(ctx, float_label, &num_values, &test_values_float);
  if (num_values != 2) printf("Incorrect num values for float: %zu != 2\n", num_values);
  if (test_values_float[0] != 21.0f) printf("Incorrect context data for float[0]: %f != 21.0\n", test_values_float[0]);
  if (test_values_float[1] != 22.0f) printf("Incorrect context data for float[1]: %f != 22.0\n", test_values_float[1]);
  CeedQFunctionContextRestoreFloatRead(ctx, float_label, &test_values_float);

  // Double field
  CeedQFunctionContextGetFieldLabel(ctx, "time", &time_label);
  double        value_time = 2.0;
  const double *test_value_time;

  CeedQFunctionContextSetDouble(ctx, time_label, &value_time);
  if (ctx_data.time != 2.0) printf("Incorrect context data for time: %f != 2.0\n", ctx_data.time);
  CeedQFunctionContextGetDoubleRead(ctx, time_label, &num_values, &test_value_time);
  if (num_values != 1) printf("Incorrect num values for time: %zu != 1\n", num_values);
  if (test_value_time[0] != 2.0) printf("Incorrect context data for time: %f != 2.0\n", test_value_time[0]);
  CeedQFunctionContextRestoreDoubleRead(ctx, time_label, &test_value_time);

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
