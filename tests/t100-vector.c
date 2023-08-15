/// @file
/// Test creation, setting, reading, restoring, and destroying of a vector
/// \test Test creation, setting, reading, restoring, and destroying of a vector

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;
  CeedScalar array[len];

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  for (CeedInt i = 0; i < len; i++) array[i] = len + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, array);

  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      if (read_array[i] != len + i) printf("Error reading array[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)read_array[i]);
    }
    CeedVectorRestoreArrayRead(x, &read_array);
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
