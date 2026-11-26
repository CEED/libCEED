/// @file
/// Test strided setting and copying of vectors
/// \test Test strided setting and copying of vectors
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedSize    start = 2, step = 3;
  CeedVector  x, y;
  CeedInt     len = 10, num_test_mem_types = 1;
  CeedMemType backend_mem_type = CEED_MEM_HOST;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &y);

  CeedGetPreferredMemType(ceed, &backend_mem_type);
  // If backend prefers DEVICE memory, then the test loops twice
  if (backend_mem_type == CEED_MEM_DEVICE) num_test_mem_types = 2;

  // Loop to test both HOST and DEVICE, as needed
  for (CeedInt m = 0; m < num_test_mem_types; m++) {
    CeedMemType test_mem_type = m == 0 ? CEED_MEM_HOST : CEED_MEM_DEVICE;

    // Initialize array for HOST sync
    CeedScalar ones[len];
    for (CeedInt i = 0; i < len; i++) ones[i] = 1.0;

    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, ones);
    {
      CeedScalar *array;
      // Sync by getting writable access to target memory, invalidating other pointer
      CeedVectorGetArray(x, test_mem_type, &array);
      CeedVectorRestoreArray(x, &array);
    }

    // Set strided
    CeedVectorSetValueStrided(x, start, -1, step, 42.0);

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
    CeedVectorCopyStrided(x, start, -1, step, y);
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
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
