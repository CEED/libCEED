/// @file
/// Test taking the reciprocal of a vector
/// \test Test taking the reciprocal of a vector

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  {
    CeedScalar array[len];

    for (CeedInt i = 0; i < len; i++) array[i] = len + i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }
  CeedVectorReciprocal(x);

  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      if (fabs(read_array[i] - 1. / (len + i)) > 10. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error taking reciprocal, array[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)read_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(x, &read_array);
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
