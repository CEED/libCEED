/// @file
/// Test creation, copying, and destroying of a vector
/// \test Test creation, copying, and destroying of a vector
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, x_copy;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &x_copy);

  {
    CeedScalar array[len], array_copy[len];

    for (CeedInt i = 0; i < len; i++) {
      array[i]      = 10 + i;
      array_copy[i] = i;
    }

    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
    CeedVectorSetArray(x_copy, CEED_MEM_HOST, CEED_COPY_VALUES, array_copy);
  }

  CeedVectorCopy(x, x_copy);

  {
    const CeedScalar *read_array;
    // Check that new array from x_copy is the same as the original input array a
    CeedVectorGetArrayRead(x_copy, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      if ((10 + i) != read_array[i]) printf("Error in copying values of CeedVector\n");
    }
    CeedVectorRestoreArrayRead(x_copy, &read_array);
  }
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&x_copy);
  CeedDestroy(&ceed);
  return 0;
}
