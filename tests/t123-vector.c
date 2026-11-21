/// @file
/// Test scaling a vector
/// \test Test scaling of a vector

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  x;
  CeedInt     len = 10, num_mem_types = 1;  // num_mem_types keeps track of both host and device side (2).
  CeedMemType preferred_type = CEED_MEM_HOST;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  // Test both host and device
  CeedGetPreferredMemType(ceed, &preferred_type);
  // If prefered memory type is device, then we need to double loop to test both sides to ensure coverage.
  if (preferred_type == CEED_MEM_DEVICE) num_mem_types = 2;

  CeedVectorCreate(ceed, len, &x);
  // Loop through each mem type to ensure that it is tested.
  for (CeedInt m = 0; m < num_mem_types; m++) {
    // First check host, then check device
    CeedMemType mem_type = m == 0 ? CEED_MEM_HOST : CEED_MEM_DEVICE;

    // Initialize array
    CeedScalar array[len];
    for (CeedInt i = 0; i < len; i++) array[i] = 10 + i;

    // Set the array, runs twice if mem_device chosen resulting in overwriting for the second test.
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);

    // Sync the array to the type we are working on before testing the operation.
    CeedVectorSyncArray(x, mem_type);

    // Scale
    CeedVectorScale(x, -0.5);

    // Test output
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
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
