/// @file
/// Test registering and setting QFunctionContext fields
/// \test Test registering and setting QFunctionContext fields
#include <ceed.h>
#include <stddef.h>
#include <string.h>

typedef struct {
  double time;
  int count;
} TestContext;

int main(int argc, char **argv) {
  Ceed ceed;
  CeedQFunctionContext ctx;
  CeedContextFieldLabel time_label, count_label;

  TestContext ctx_data = {
    .time = 1.0,
    .count = 42
  };

  CeedInit(argv[1], &ceed);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof(TestContext), &ctx_data);

  CeedQFunctionContextRegisterDouble(ctx, "time", offsetof(TestContext, time),
                                     "current time");
  CeedQFunctionContextRegisterInt32(ctx, "count", offsetof(TestContext, count),
                                    "some sort of counter");

  const CeedContextFieldLabel *field_labels;
  CeedInt num_fields;
  CeedQFunctionContextGetAllFieldLabels(ctx, &field_labels, &num_fields);
  if (num_fields != 2)
    // LCOV_EXCL_START
    printf("Incorrect number of fields set: %d != 2", num_fields);
  // LCOV_EXCL_STOP
  const char *name;
  CeedContextFieldType type;
  CeedContextFieldLabelGetDescription(field_labels[0], &name, NULL, &type);
  if (strcmp(name, "time"))
    // LCOV_EXCL_START
    printf("Incorrect context field description for time: \"%s\" != \"time\"",
           name);
  // LCOV_EXCL_STOP
  if (type != CEED_CONTEXT_FIELD_DOUBLE)
    // LCOV_EXCL_START
    printf("Incorrect context field type for time: \"%s\" != \"%s\"",
           CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_DOUBLE]);
  // LCOV_EXCL_STOP
  CeedContextFieldLabelGetDescription(field_labels[1], &name, NULL, &type);
  if (strcmp(name, "count"))
    // LCOV_EXCL_START
    printf("Incorrect context field description for count: \"%s\" != \"count\"",
           name);
  // LCOV_EXCL_STOP
  if (type != CEED_CONTEXT_FIELD_INT32)
    // LCOV_EXCL_START
    printf("Incorrect context field type for count: \"%s\" != \"%s\"",
           CeedContextFieldTypes[type], CeedContextFieldTypes[CEED_CONTEXT_FIELD_INT32]);
  // LCOV_EXCL_STOP

  CeedQFunctionContextGetFieldLabel(ctx, "time", &time_label);
  CeedQFunctionContextSetDouble(ctx, time_label, 2.0);
  if (ctx_data.time != 2.0)
    // LCOV_EXCL_START
    printf("Incorrect context data for time: %f != 2.0", ctx_data.time);
  // LCOV_EXCL_STOP

  CeedQFunctionContextGetFieldLabel(ctx, "count", &count_label);
  CeedQFunctionContextSetInt32(ctx, count_label, 43);
  if (ctx_data.count != 43)
    // LCOV_EXCL_START
    printf("Incorrect context data for count: %d != 43", ctx_data.count);
  // LCOV_EXCL_STOP

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
