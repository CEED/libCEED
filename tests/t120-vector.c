/// @file
/// Test creation, copying, and destroying of a vector
/// \test Test creation, copying, and destroying of a vector
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, x_2;
  CeedInt    n;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorCreate(ceed, n + 1, &x_2);

  CeedVectorReferenceCopy(x, &x_2);  // This destroys the previous x_2
  CeedVectorDestroy(&x);

  CeedSize len;
  CeedVectorGetLength(x_2, &len);  // Second reference still valid
  if (len != n) printf("Error copying CeedVector reference\n");

  CeedVectorDestroy(&x_2);
  CeedDestroy(&ceed);
  return 0;
}
