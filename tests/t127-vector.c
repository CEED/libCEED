/// @file
/// Test strided setting and copying of vectors
/// \test Test strided setting and copying of vectors
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedSize   start = 2, step = 3;
  CeedVector x, y;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &y);
  
  // Set strided
  CeedVectorSetValue(x, 1.0);
  CeedVectorSetValueStrided(x, start, step, 42.0);
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      CeedScalar value = (i - start) % step == 0 ? 42.0 : 1.0;

      if (read_array[i] != value) {
        // LCOV_EXCL_START
        printf("Error in setting value in x at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], value);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(x, &read_array);
  }
  
  // Copy strided
  CeedVectorSetValue(y, 0.0);
  CeedVectorCopyStrided(x, start, step, y);
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      CeedScalar value = (i - start) % step == 0 ? 42.0 : 0.0;

      if (read_array[i] != value) {
        // LCOV_EXCL_START
        printf("Error in copying value to y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], value);
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
