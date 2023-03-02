/// @file
/// Test viewing a vector
/// \test Test viewing a vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;
  CeedScalar array[len];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  for (CeedInt i = 0; i < len; i++) array[i] = len + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, array);

  CeedVectorView(x, "%12.8f", stdout);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
