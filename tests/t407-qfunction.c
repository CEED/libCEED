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
  int32_t    count[2];
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
  CeedContextFieldLabel count_label, int64_label, size_label, scalar_label;
  CeedContextFieldLabel float_label, time_label;

  TestContext ctx_data = {
      .is_set       = true,
      .byte_value   = {1,    2   },
      .int8_value   = {2,    3   },
      .int_value    = {3,    4   },
      .count        = {13,   42  },
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
  CeedQFunctionContextRegisterInt32(ctx, "count", offsetof(TestContext, count), 2, "some sort of counter");
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
  if (strcmp(name, "is set")) printf("Incorrect context field description for is set: \"%s\" != \"is set\"\n", name);
  if (num_values != 1) printf("Incorrect context field number of values for is set: \"%zu\" != 1\n", num_values);
  if (type != CEED_CONTEXT_FIELD_BOOL) {
    printf("Incorrect context field type for is set: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_BOOL]);
  }

  CeedContextFieldLabelGetDescription(field_labels[1], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "byte")) printf("Incorrect context field description for byte: \"%s\" != \"byte\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for byte: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_BYTE) {
    printf("Incorrect context field type for byte: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_BYTE]);
  }

  CeedContextFieldLabelGetDescription(field_labels[2], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "int8")) printf("Incorrect context field description for int8: \"%s\" != \"int8\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for int8: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT8) {
    printf("Incorrect context field type for int8: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT8]);
  }

  CeedContextFieldLabelGetDescription(field_labels[3], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "int")) printf("Incorrect context field description for int: \"%s\" != \"int\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for int: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT) {
    printf("Incorrect context field type for int: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT]);
  }

  CeedContextFieldLabelGetDescription(field_labels[4], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "count")) printf("Incorrect context field description for count: \"%s\" != \"count\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for count: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT32) {
    printf("Incorrect context field type for count: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT32]);
  }

  CeedContextFieldLabelGetDescription(field_labels[5], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "int64")) printf("Incorrect context field description for int64: \"%s\" != \"int64\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for int64: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT64) {
    printf("Incorrect context field type for int64: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT64]);
  }

  CeedContextFieldLabelGetDescription(field_labels[6], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "size")) printf("Incorrect context field description for size: \"%s\" != \"size\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for size: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_SIZE) {
    printf("Incorrect context field type for size: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_SIZE]);
  }

  CeedContextFieldLabelGetDescription(field_labels[7], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "scalar")) printf("Incorrect context field description for scalar: \"%s\" != \"scalar\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for scalar: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_SCALAR) {
    printf("Incorrect context field type for scalar: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_SCALAR]);
  }

  CeedContextFieldLabelGetDescription(field_labels[8], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "float")) printf("Incorrect context field description for float: \"%s\" != \"float\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for float: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_FLOAT) {
    printf("Incorrect context field type for float: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_FLOAT]);
  }

  CeedContextFieldLabelGetDescription(field_labels[9], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "time")) printf("Incorrect context field description for time: \"%s\" != \"time\"\n", name);
  if (num_values != 1) printf("Incorrect context field number of values for time: \"%zu\" != 1\n", num_values);
  if (type != CEED_CONTEXT_FIELD_DOUBLE) {
    printf("Incorrect context field type for time: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_DOUBLE]);
  }

  // Boolean field
  CeedQFunctionContextGetFieldLabel(ctx, "is set", &is_set_label);
  bool value_is_set = false;

  CeedQFunctionContextSetBoolean(ctx, is_set_label, &value_is_set);
  if (ctx_data.is_set != false) printf("Incorrect context data for is_set: %d != 0\n", ctx_data.is_set);

  // Byte field
  CeedQFunctionContextGetFieldLabel(ctx, "byte", &byte_label);
  char values_byte[2] = {8, 9};

  CeedQFunctionContextSetByte(ctx, byte_label, values_byte);
  if (ctx_data.byte_value[0] != 8) printf("Incorrect context data for byte[0]: %d != 8\n", ctx_data.byte_value[0]);
  if (ctx_data.byte_value[1] != 9) printf("Incorrect context data for byte[1]: %d != 9\n", ctx_data.byte_value[1]);

  // CeedInt8 field
  CeedQFunctionContextGetFieldLabel(ctx, "int8", &int8_label);
  CeedInt8 values_int8[2] = {10, 11};

  CeedQFunctionContextSetCeedInt8(ctx, int8_label, values_int8);
  if (ctx_data.int8_value[0] != 10) printf("Incorrect context data for int8[0]: %" CeedInt8_FMT " != 10\n", ctx_data.int8_value[0]);
  if (ctx_data.int8_value[1] != 11) printf("Incorrect context data for int8[1]: %" CeedInt8_FMT " != 11\n", ctx_data.int8_value[1]);

  // CeedInt field
  CeedQFunctionContextGetFieldLabel(ctx, "int", &int_label);
  CeedInt values_int[2] = {12, 13};

  CeedQFunctionContextSetCeedInt(ctx, int_label, values_int);
  if (ctx_data.int_value[0] != 12) printf("Incorrect context data for int[0]: %" CeedInt_FMT " != 12\n", ctx_data.int_value[0]);
  if (ctx_data.int_value[1] != 13) printf("Incorrect context data for int[1]: %" CeedInt_FMT " != 13\n", ctx_data.int_value[1]);

  // Int32 field
  CeedQFunctionContextGetFieldLabel(ctx, "count", &count_label);
  int32_t values_count[2] = {14, 43};

  CeedQFunctionContextSetInt32(ctx, count_label, values_count);
  if (ctx_data.count[0] != 14) printf("Incorrect context data for count[0]: %" CeedInt_FMT " != 14\n", ctx_data.count[0]);
  if (ctx_data.count[1] != 43) printf("Incorrect context data for count[1]: %" CeedInt_FMT " != 43\n", ctx_data.count[1]);

  // Int64 field
  CeedQFunctionContextGetFieldLabel(ctx, "int64", &int64_label);
  int64_t values_int64[2] = {15, 16};

  CeedQFunctionContextSetInt64(ctx, int64_label, values_int64);
  if (ctx_data.int64_value[0] != 15) printf("Incorrect context data for int64[0]: %" PRId64 " != 15\n", ctx_data.int64_value[0]);
  if (ctx_data.int64_value[1] != 16) printf("Incorrect context data for int64[1]: %" PRId64 " != 16\n", ctx_data.int64_value[1]);

  // CeedSize field
  CeedQFunctionContextGetFieldLabel(ctx, "size", &size_label);
  CeedSize values_size[2] = {17, 18};

  CeedQFunctionContextSetCeedSize(ctx, size_label, values_size);
  if (ctx_data.size_value[0] != 17) printf("Incorrect context data for size[0]: %" CeedSize_FMT " != 17\n", ctx_data.size_value[0]);
  if (ctx_data.size_value[1] != 18) printf("Incorrect context data for size[1]: %" CeedSize_FMT " != 18\n", ctx_data.size_value[1]);

  // CeedScalar field
  CeedQFunctionContextGetFieldLabel(ctx, "scalar", &scalar_label);
  CeedScalar values_scalar[2] = {19.0, 20.0};

  CeedQFunctionContextSetCeedScalar(ctx, scalar_label, values_scalar);
  if (ctx_data.scalar_value[0] != (CeedScalar)19.0) printf("Incorrect context data for scalar[0]: %g != 19.0\n", (double)ctx_data.scalar_value[0]);
  if (ctx_data.scalar_value[1] != (CeedScalar)20.0) printf("Incorrect context data for scalar[1]: %g != 20.0\n", (double)ctx_data.scalar_value[1]);

  // Float field
  CeedQFunctionContextGetFieldLabel(ctx, "float", &float_label);
  float values_float[2] = {21.0f, 22.0f};

  CeedQFunctionContextSetFloat(ctx, float_label, values_float);
  if (ctx_data.float_value[0] != 21.0f) printf("Incorrect context data for float[0]: %f != 21.0\n", ctx_data.float_value[0]);
  if (ctx_data.float_value[1] != 22.0f) printf("Incorrect context data for float[1]: %f != 22.0\n", ctx_data.float_value[1]);

  // Double field
  CeedQFunctionContextGetFieldLabel(ctx, "time", &time_label);
  double value_time = 2.0;

  CeedQFunctionContextSetDouble(ctx, time_label, &value_time);
  if (ctx_data.time != 2.0) printf("Incorrect context data for time: %f != 2.0\n", ctx_data.time);

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
