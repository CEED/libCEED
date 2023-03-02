/// @file
/// Test syncing device data to host pointer
/// \test Test syncing device data to host pointer
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, y;
  CeedInt    len = 10;
  CeedScalar x_array[len];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &y);

  for (CeedInt i = 0; i < len; i++) x_array[i] = 0;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  {
    CeedScalar initial_array[len];

    for (CeedInt i = 0; i < len; i++) initial_array[i] = len + i;
    CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, initial_array);
  }
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(y, CEED_MEM_DEVICE, &read_array);
    CeedVectorSetArray(x, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)read_array);
    CeedVectorRestoreArrayRead(y, &read_array);
  }
  CeedVectorSyncArray(x, CEED_MEM_HOST);
  for (CeedInt i = 0; i < len; i++) {
    if (x_array[i] != len + i) printf("Error reading array[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)x_array[i]);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
