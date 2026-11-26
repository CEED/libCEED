/// @file
/// Test syncing device data to host pointer and host data to device pointer
/// \test Test syncing device data to host pointer and host data to device pointer
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  y;
  CeedInt     len = 10, num_test_mem_types = 1;
  CeedMemType backend_mem_type = CEED_MEM_HOST;

  CeedInit(argv[1], &ceed);

  CeedGetPreferredMemType(ceed, &backend_mem_type);
  // If backend prefers DEVICE memory, then the test loops twice
  if (backend_mem_type == CEED_MEM_DEVICE) num_test_mem_types = 2;

  CeedVectorCreate(ceed, len, &y);
  {
    CeedScalar initial_array[len];

    for (CeedInt i = 0; i < len; i++) initial_array[i] = len + i;
    CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, initial_array);
  }

  // Loop to test both HOST and DEVICE, as needed
  for (CeedInt m = 0; m < num_test_mem_types; m++) {
    // First sync to a borrowed host array, then to a borrowed device array
    CeedMemType test_mem_type  = m == 0 ? CEED_MEM_HOST : CEED_MEM_DEVICE;
    CeedMemType other_mem_type = m == 0 ? CEED_MEM_DEVICE : CEED_MEM_HOST;
    CeedVector  x, x_owner                      = NULL;
    CeedScalar  x_array[len], *x_borrowed_array = x_array;

    CeedVectorCreate(ceed, len, &x);

    // Borrow an array in the memory type being tested
    for (CeedInt i = 0; i < len; i++) x_array[i] = 0;
    if (test_mem_type == CEED_MEM_DEVICE) {
      CeedScalar *owner_array;

      // Tests cannot allocate device memory directly, so borrow zeroed device memory owned by another vector
      CeedVectorCreate(ceed, len, &x_owner);
      CeedVectorSetValue(x_owner, 0.0);
      CeedVectorGetArray(x_owner, CEED_MEM_DEVICE, &owner_array);
      x_borrowed_array = owner_array;
      CeedVectorRestoreArray(x_owner, &owner_array);
    }
    CeedVectorSetArray(x, test_mem_type, CEED_USE_POINTER, x_borrowed_array);

    // Set values in the other memory type, leaving the borrowed array invalid
    {
      const CeedScalar *read_array;

      CeedVectorGetArrayRead(y, other_mem_type, &read_array);
      CeedVectorSetArray(x, other_mem_type, CEED_COPY_VALUES, (CeedScalar *)read_array);
      CeedVectorRestoreArrayRead(y, &read_array);
    }

    // Sync back to the memory type being tested, which should copy into the borrowed array
    CeedVectorSyncArray(x, test_mem_type);

    // Check the borrowed array; device memory is read back through the vector that owns it
    {
      const CeedScalar *check_array = x_array;

      if (test_mem_type == CEED_MEM_DEVICE) CeedVectorGetArrayRead(x_owner, CEED_MEM_HOST, &check_array);
      for (CeedInt i = 0; i < len; i++) {
        if (check_array[i] != len + i) printf("Error reading array[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)check_array[i]);
      }
      if (test_mem_type == CEED_MEM_DEVICE) CeedVectorRestoreArrayRead(x_owner, &check_array);
    }

    CeedVectorDestroy(&x);
    CeedVectorDestroy(&x_owner);
  }

  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
