/// @file
/// Test registering and setting QFunctionContext fields
/// \test Test registering and setting QFunctionContext fields
#include <ceed.h>
#include <stddef.h>
#include <string.h>

typedef struct {
  double time;
  int count[2];
} TestContext;

int main(int argc, char **argv) {
  Ceed ceed;
  CeedQFunctionContext ctx;
  CeedContextFieldLabel time_label, count_label;

  TestContext ctx_data = {
    .time = 1.0,
    .count = {13, 42},
  };

  CeedInit(argv[1], &ceed);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof(TestContext), &ctx_data);

  CeedQFunctionContextRegisterDouble(ctx, "time", offsetof(TestContext, time),
                                     1, "current time");
  CeedQFunctionContextRegisterInt32(ctx, "count", offsetof(TestContext, count),
                                    2, "some sort of counter");

  const CeedContextFieldLabel *field_labels;
  CeedInt num_fields;
  CeedQFunctionContextGetAllFieldLabels(ctx, &field_labels, &num_fields);
  if (num_fields != 2)
    // LCOV_EXCL_START
    printf("Incorrect number of fields set: %d != 2\n", num_fields);
  // LCOV_EXCL_STOP

  const char *name;
  size_t num_values;
  CeedContextFieldType type;
  CeedContextFieldLabelGetDescription(field_labels[0], &name, NULL, &num_values,
                                      &type);
  if (strcmp(name, "time"))
    // LCOV_EXCL_START
    printf("Incorrect context field description for time: \"%s\" != \"time\"\n",
           name);
  // LCOV_EXCL_STOP
  if (num_values != 1)
    // LCOV_EXCL_START
    printf("Incorrect context field number of values for time: \"%ld\" != 1\n",
           num_values);
  // LCOV_EXCL_STOP
  if (type != CEED_CONTEXT_FIELD_DOUBLE)
    // LCOV_EXCL_START
    printf("Incorrect context field type for time: \"%s\" != \"%s\"\n",
           CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_DOUBLE]);
  // LCOV_EXCL_STOP

  CeedContextFieldLabelGetDescription(field_labels[1], &name, NULL, &num_values,
                                      &type);
  if (strcmp(name, "count"))
    // LCOV_EXCL_START
    printf("Incorrect context field description for count: \"%s\" != \"count\"\n",
           name);
  // LCOV_EXCL_STOP
  if (num_values != 2)
    // LCOV_EXCL_START
    printf("Incorrect context field number of values for count: \"%ld\" != 2\n",
           num_values);
  // LCOV_EXCL_STOP
  if (type != CEED_CONTEXT_FIELD_INT32)
    // LCOV_EXCL_START
    printf("Incorrect context field type for count: \"%s\" != \"%s\"\n",
           CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT32]);
  // LCOV_EXCL_STOP

  CeedQFunctionContextGetFieldLabel(ctx, "time", &time_label);
  double value_time = 2.0;
  CeedQFunctionContextSetDouble(ctx, time_label, &value_time);
  if (ctx_data.time != 2.0)
    // LCOV_EXCL_START
    printf("Incorrect context data for time: %f != 2.0\n", ctx_data.time);
  // LCOV_EXCL_STOP

  CeedQFunctionContextGetFieldLabel(ctx, "count", &count_label);
  int values_count[2] = {14, 43};
  CeedQFunctionContextSetInt32(ctx, count_label, (int *)&values_count);
  if (ctx_data.count[0] != 14)
    // LCOV_EXCL_START
    printf("Incorrect context data for count[0]: %d != 14\n", ctx_data.count[0]);
  // LCOV_EXCL_STOP
  if (ctx_data.count[1] != 43)
    // LCOV_EXCL_START
    printf("Incorrect context data for count[1]: %d != 43\n", ctx_data.count[1]);
  // LCOV_EXCL_STOP

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
