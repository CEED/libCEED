/// @file
/// Test registering and setting QFunctionContext fields
/// \test Test registering and setting QFunctionContext fields
#include <ceed.h>
#include <ceed/backend.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  double time;
  int    count[2];
  bool   is_set;
} TestContext;

int main(int argc, char **argv) {
  Ceed                  ceed;
  CeedQFunctionContext  ctx;
  CeedContextFieldLabel time_label, count_label, is_set_label;

  TestContext ctx_data = {
      .time   = 1.0,
      .count  = {13, 42},
      .is_set = true,
  };

  CeedInit(argv[1], &ceed);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(TestContext), &ctx_data);

  CeedQFunctionContextRegisterDouble(ctx, "time", offsetof(TestContext, time), 1, "current time");
  CeedQFunctionContextRegisterInt32(ctx, "count", offsetof(TestContext, count), 2, "some sort of counter");
  CeedQFunctionContextRegisterBoolean(ctx, "is set", offsetof(TestContext, is_set), 1, "some boolean flag");

  const CeedContextFieldLabel *field_labels;
  CeedInt                      num_fields;
  CeedQFunctionContextGetAllFieldLabels(ctx, &field_labels, &num_fields);
  if (num_fields != 3) printf("Incorrect number of fields set: %" CeedInt_FMT " != 2\n", num_fields);

  const char          *name;
  size_t               num_values;
  CeedContextFieldType type;

  CeedContextFieldLabelGetDescription(field_labels[0], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "time")) printf("Incorrect context field description for time: \"%s\" != \"time\"\n", name);
  if (num_values != 1) printf("Incorrect context field number of values for time: \"%zu\" != 1\n", num_values);
  if (type != CEED_CONTEXT_FIELD_DOUBLE) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for time: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_DOUBLE]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[1], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "count")) printf("Incorrect context field description for count: \"%s\" != \"count\"\n", name);
  if (num_values != 2) printf("Incorrect context field number of values for count: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_INT32) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for count: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT32]);
    // LCOV_EXCL_STOP
  }

  CeedContextFieldLabelGetDescription(field_labels[2], &name, NULL, &num_values, NULL, &type);
  if (strcmp(name, "is set")) printf("Incorrect context field description for count: \"%s\" != \"count\"\n", name);
  if (num_values != 1) printf("Incorrect context field number of values for count: \"%zu\" != 2\n", num_values);
  if (type != CEED_CONTEXT_FIELD_BOOL) {
    // LCOV_EXCL_START
    printf("Incorrect context field type for is_set: \"%s\" != \"%s\"\n", CeedContextFieldTypes[type],
           CeedContextFieldTypes[CEED_CONTEXT_FIELD_BOOL]);
    // LCOV_EXCL_STOP
  }

  CeedQFunctionContextGetFieldLabel(ctx, "time", &time_label);
  double value_time = 2.0;

  CeedQFunctionContextSetDouble(ctx, time_label, &value_time);
  if (ctx_data.time != 2.0) printf("Incorrect context data for time: %f != 2.0\n", ctx_data.time);

  CeedQFunctionContextGetFieldLabel(ctx, "count", &count_label);
  int values_count[2] = {14, 43};

  CeedQFunctionContextSetInt32(ctx, count_label, (int *)&values_count);
  if (ctx_data.count[0] != 14) printf("Incorrect context data for count[0]: %" CeedInt_FMT " != 14\n", ctx_data.count[0]);
  if (ctx_data.count[1] != 43) printf("Incorrect context data for count[1]: %" CeedInt_FMT " != 43\n", ctx_data.count[1]);

  CeedQFunctionContextGetFieldLabel(ctx, "is set", &is_set_label);
  bool value_is_set = false;

  CeedQFunctionContextSetBoolean(ctx, is_set_label, &value_is_set);
  if (ctx_data.is_set != false) printf("Incorrect context data for is_set: %d != true\n", ctx_data.is_set);

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
