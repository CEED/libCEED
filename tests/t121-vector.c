/// @file
/// Test summing of a pair of vectors
/// \test Test summing of a pair of vectors

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  x, y;
  CeedInt     len = 10, num_test_mem_types = 1;
  CeedMemType backend_mem_type = CEED_MEM_HOST;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedGetPreferredMemType(ceed, &backend_mem_type);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &y);

  // If backend prefers DEVICE memory, then the test loops twice
  if (backend_mem_type == CEED_MEM_DEVICE) num_test_mem_types = 2;

  // Loop to test both HOST and DEVICE, as needed
  for (CeedInt m = 0; m < num_test_mem_types; m++) {
    // First check Host, then Device when syncing array.
    CeedMemType test_mem_type = m == 0 ? CEED_MEM_HOST : CEED_MEM_DEVICE;

    CeedScalar array[len];

    for (CeedInt i = 0; i < len; i++) array[i] = 10 + i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
    CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, array);

    {
      CeedScalar *array;
      // Sync by getting writable access to target memory, invalidating other pointer
      CeedVectorGetArray(y, test_mem_type, &array);
      CeedVectorRestoreArray(y, &array);
    }
    CeedVectorAXPY(y, -0.5, x);

    {
      const CeedScalar *read_array;

      CeedVectorGetArrayRead(y, CEED_MEM_HOST, &read_array);
      for (CeedInt i = 0; i < len; i++) {
        if (fabs(read_array[i] - (10.0 + i) / 2) > 1e-14) {
          // LCOV_EXCL_START
          printf("Error in alpha x + y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], (10.0 + i) / 2);
          // LCOV_EXCL_STOP
        }
      }
      CeedVectorRestoreArrayRead(y, &read_array);
    }
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
