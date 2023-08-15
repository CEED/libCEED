/// @file
/// Test CeedVectorGetArrayWrite to modify array
/// \test Test CeedVectorGetArrayWrite to modify array

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
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
    CeedScalar *writable_array;

    CeedVectorGetArrayWrite(x, CEED_MEM_HOST, &writable_array);
    for (CeedInt i = 0; i < len; i++) writable_array[i] = 3 * i;
    CeedVectorRestoreArray(x, &writable_array);
  }
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, (const CeedScalar **)&read_array);
    for (CeedInt i = 0; i < len; i++) {
      if (read_array[i] != (CeedScalar)(3 * i)) printf("Error writing array[%" CeedInt_FMT "] = %f\n", i, read_array[i]);
    }
    CeedVectorRestoreArrayRead(x, (const CeedScalar **)&read_array);
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
