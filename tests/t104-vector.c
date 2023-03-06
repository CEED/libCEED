/// @file
/// Test CeedVectorGetArray to modify array
/// \test Test CeedVectorGetArray to modify array
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    x;
  const CeedInt len = 10;
  CeedScalar    array[len];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  for (CeedInt i = 0; i < len; i++) array[i] = 0;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, array);

  {
    CeedScalar *writable_array;

    CeedVectorGetArray(x, CEED_MEM_HOST, &writable_array);
    writable_array[3] = -3.14;
    CeedVectorRestoreArray(x, &writable_array);
  }

  if (array[3] != (CeedScalar)(-3.14)) printf("Error writing array[3] = %f\n", (CeedScalar)array[3]);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
