/// @file
/// Test CeedVectorGetArrayWrite to modify array
/// \test Test CeedVectorGetArrayWrite to modify array
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    x;
  const CeedInt len = 10;

  CeedInit(argv[1], &ceed);

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
