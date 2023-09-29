/// @file
/// Test creation, evaluation, and destruction for QFunction
/// \test Test creation, evaluation, and destruction for QFunction
#include "t409-qfunction.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                 ceed;
  CeedVector           in[16], out[16];
  CeedVector           u, v;
  CeedQFunction        qf;
  CeedQFunctionContext ctx;
  CeedInt              q           = 8;
  bool                 is_writable = true;
  CeedScalar           ctx_data[5] = {1, 2, 3, 4, 5};

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, q, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, q, &v);
  CeedVectorSetValue(v, 0.0);

  CeedQFunctionCreateInterior(ceed, 1, scale, scale_loc, &qf);
  CeedQFunctionAddInput(qf, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf, "v", 1, CEED_EVAL_INTERP);

  CeedQFunctionContextCreate(ceed, &ctx);
  CeedQFunctionContextSetData(ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(ctx_data), &ctx_data);
  CeedQFunctionSetContext(qf, ctx);
  CeedQFunctionSetContextWritable(qf, is_writable);
  {
    in[0]  = u;
    out[0] = v;
    CeedQFunctionApply(qf, q, in, out);
  }

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < q; i++) {
      if (fabs(v_array[i] - ctx_data[1]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("v[%" CeedInt_FMT "] %f != 2.0\n", i, v_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  // Check for written context data
  CeedScalar *ctx_data_new;
  CeedQFunctionContextGetDataRead(ctx, CEED_MEM_HOST, &ctx_data_new);
  if (ctx_data_new[0] != 42) {
    // LCOV_EXCL_START
    printf("Context data not written: %f != 42\n", ctx_data_new[0]);
    // LCOV_EXCL_STOP
  }
  CeedQFunctionContextRestoreDataRead(ctx, &ctx_data_new);

  // Assert that context will not be written
  // Note: The interface cannot enforce this in user code
  //   so setting is_writable == false and then calling
  //   CeedQFunctionApply to mutate the context would lead
  //   to inconsistent data on the GPU.
  //   Only the `/cpu/self/memcheck/*` backends verify that
  //   read-only access resulted in no changes to the context data
  CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctx_data_new);
  ctx_data_new[0] = 5;
  CeedQFunctionContextRestoreData(ctx, &ctx_data_new);
  is_writable = false;
  CeedQFunctionSetContextWritable(qf, is_writable);
  {
    in[0]  = u;
    out[0] = v;
    // Will only error in `/cpu/self/memcheck/*` backends
    CeedQFunctionApply(qf, q, in, out);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedQFunctionDestroy(&qf);
  CeedQFunctionContextDestroy(&ctx);
  CeedDestroy(&ceed);
  return 0;
}
