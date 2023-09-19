/// @file
/// Test creation, setting, reading, restoring, and destroying of a vector using CEED_MEM_DEVICE
/// \test Test creation, setting, reading, restoring, and destroying of a vector using CEED_MEM_DEVICE
#include <ceed.h>
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

    for (CeedInt i = 0; i < len; i++) array[i] = len + i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_DEVICE, &read_array);
    CeedVectorSetArray(y, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)read_array);
    CeedVectorRestoreArrayRead(x, &read_array);
  }
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      if (read_array[i] != len + i) printf("Error reading array[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)read_array[i]);
    }
    CeedVectorRestoreArrayRead(y, &read_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
