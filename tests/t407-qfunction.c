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

  const CeedQFunctionContextFieldDescription *field_descriptions;
  CeedInt num_fields;
  CeedQFunctionContextGetFieldDescriptions(ctx, &field_descriptions, &num_fields);
  if (num_fields != 2)
    // LCOV_EXCL_START
    printf("Incorrect number of fields set: %d != 2", num_fields);
  // LCOV_EXCL_STOP
  if (strcmp(field_descriptions[0].name, "time"))
    // LCOV_EXCL_START
    printf("Incorrect context field description for time: \"%s\" != \"time\"",
           field_descriptions[0].name);
  // LCOV_EXCL_STOP
  if (strcmp(field_descriptions[1].name, "count"))
    // LCOV_EXCL_START
    printf("Incorrect context field description for time: \"%s\" != \"count\"",
           field_descriptions[1].name);
  // LCOV_EXCL_STOP

  CeedQFunctionContextSetDouble(ctx, "time", 2.0);
  if (ctx_data.time != 2.0)
    // LCOV_EXCL_START
    printf("Incorrect context data for time: %f != 2.0", ctx_data.time);
  // LCOV_EXCL_STOP

  CeedQFunctionContextSetInt32(ctx, "count", 43);
  if (ctx_data.count != 43)
    // LCOV_EXCL_START
    printf("Incorrect context data for time: %d != 43", ctx_data.count);
  // LCOV_EXCL_STOP

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
