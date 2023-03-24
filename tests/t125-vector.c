/// @file
/// Test summing of a pair of vectors
/// \test Test y = alpha x + beta y
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, y;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &y);
  {
    CeedScalar array[len];

    for (CeedInt i = 0; i < len; i++) array[i] = 10 + i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
    CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(y, type);
  }
  CeedVectorAXPBY(y, -0.5, 1.0, x);

  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      if (fabs(read_array[i] - (10.0 + i) * 1.5) > 1e-14) {
        // LCOV_EXCL_START
        printf("Error in alpha x + y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], (10.0 + i) * 1.5);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(y, &read_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
