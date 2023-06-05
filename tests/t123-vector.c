/// @file
/// Test scaling a vector
/// \test Test scaling of a vector
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  {
    CeedScalar array[len];

    for (CeedInt i = 0; i < len; i++) array[i] = 10 + i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }
  CeedVectorScale(x, -0.5);

  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      if (fabs(read_array[i] + (10.0 + i) / 2) > 1e-14) {
        // LCOV_EXCL_START
        printf("Error in alpha x at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], -(10.0 + i) / 2);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(x, &read_array);
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
