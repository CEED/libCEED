/// @file
/// Test CeedVectorSetArray to remove array access
/// \test Test CeedVectorSetArray to remove array access
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    x;
  const CeedInt len = 10;
  CeedScalar    array[len];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  for (CeedInt i = 0; i < len; i++) array[i] = 0;
  array[3] = -3.14;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, array);

  // Taking array should return a
  {
    CeedScalar *array;

    CeedVectorTakeArray(x, CEED_MEM_HOST, &array);
    if (fabs(array[3] + 3.14) > 10. * CEED_EPSILON) printf("Error taking array, array[3] = %f\n", (CeedScalar)array[3]);
  }

  // Getting array should not modify a
  {
    CeedScalar *writable_array;

    CeedVectorGetArrayWrite(x, CEED_MEM_HOST, &writable_array);
    writable_array[5] = -3.14;
    CeedVectorRestoreArray(x, &writable_array);
  }
  if (fabs(array[5] + 3.14) < 10. * CEED_EPSILON) printf("Error protecting array, array[3] = %f\n", (CeedScalar)array[3]);

  CeedVectorDestroy(&x);

  // Test with a size zero vector
  CeedVectorCreate(ceed, 0, &x);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, NULL);
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    if (read_array) printf("CeedVectorGetArrayRead returned non-NULL for zero-sized Vector\n");
    CeedVectorRestoreArrayRead(x, &read_array);
  }
  {
    CeedScalar *array;

    CeedVectorTakeArray(x, CEED_MEM_HOST, &array);
    if (array) printf("CeedVectorTakeArray returned non-NULL for zero-sized Vector\n");
  }
  CeedVectorDestroy(&x);

  CeedDestroy(&ceed);
  return 0;
}
