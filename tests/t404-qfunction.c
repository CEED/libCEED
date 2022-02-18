/// @file
/// Test creation, setting, and taking data for QFunctionContext
/// \test Test creation, setting, and taking data for QFunctionContext
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedQFunctionContext ctx;
  CeedScalar ctxData[5] = {1, 2, 3, 4, 5}, *ctxDataCopy;

  CeedInit(argv[1], &ceed);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof(ctxData), &ctxData);

  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctxDataCopy);
  ctxDataCopy[4] = 6;
  CeedQFunctionContextRestoreData(ctx, &ctxDataCopy);
  if (ctxData[4] != 6)
    // LCOV_EXCL_START
    printf("error modifying data: %f != 6.0\n", ctxData[4]);
  // LCOV_EXCL_STOP

  CeedQFunctionContextTakeData(ctx, CEED_MEM_HOST, &ctxDataCopy);
  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
