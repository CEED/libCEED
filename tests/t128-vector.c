/// @file
/// Test copying into vector with borrowed pointer
/// \test Test copying into vector with borrowed pointer
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, x_copy;
  CeedInt    len = 10;
  CeedScalar array_borrowed[len];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &x_copy);

  {
    CeedScalar array[len];

    for (CeedInt i = 0; i < len; i++) {
      array[i]          = i;
      array_borrowed[i] = 10 + i;
    }

    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
    CeedVectorSetArray(x_copy, CEED_MEM_HOST, CEED_USE_POINTER, array_borrowed);
  }

  // Copy to device if preferred
  {
    CeedMemType mem_type = CEED_MEM_HOST;

    CeedGetPreferredMemType(ceed, &mem_type);
    if (mem_type == CEED_MEM_DEVICE) CeedVectorSyncArray(x, CEED_MEM_DEVICE);
  }

  // Copy and sync borrowed array
  CeedVectorCopy(x, x_copy);
  CeedVectorSyncArray(x_copy, CEED_MEM_HOST);

  // Check that borrowed array is the same as the original input array a
  for (CeedInt i = 0; i < len; i++) {
    if (array_borrowed[i] != i) printf("Error in copying values of CeedVector\n");
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&x_copy);
  CeedDestroy(&ceed);
  return 0;
}
