/// @file
/// Test creation, setting, and taking data for QFunctionContext
/// \test Test creation, setting, and taking data for QFunctionContext
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed                 ceed;
  CeedQFunctionContext ctx;
  CeedScalar           ctxData[5] = {1, 2, 3, 4, 5}, *ctxDataCopy;

  CeedInit(argv[1], &ceed);

  // Set borrowed pointer
  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(ctxData), &ctxData);

  // Update borrowed pointer
  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctxDataCopy);
  ctxDataCopy[4] = 6;
  CeedQFunctionContextRestoreData(ctx, &ctxDataCopy);
  if (ctxData[4] != 6) printf("error modifying data: %f != 6.0\n", ctxData[4]);

  // Take back borrowed pointer
  CeedQFunctionContextTakeData(ctx, CEED_MEM_HOST, &ctxDataCopy);
  if (ctxDataCopy[4] != 6) printf("error accessing borrowed data: %f != 6.0\n", ctxDataCopy[4]);

  // Set copied data
  ctxData[4] = 6;
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctxData), &ctxData);

  // Check copied data
  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctxDataCopy);
  if (ctxDataCopy[4] != 6) printf("error accessing copied data: %f != 6.0\n", ctxDataCopy[4]);
  CeedQFunctionContextRestoreData(ctx, &ctxDataCopy);

  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
