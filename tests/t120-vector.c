/// @file
/// Test creation, reference copying, and destroying of a vector
/// \test Test creation, reference copying, and destroying of a vector

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, x_2;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len + 1, &x_2);

  CeedVectorReferenceCopy(x, &x_2);  // This destroys the previous x_2
  CeedVectorDestroy(&x);

  {
    CeedSize len_2;

    CeedVectorGetLength(x_2, &len_2);  // Second reference still valid
    if (len_2 != len) printf("Error copying CeedVector reference\n");
  }

  CeedVectorDestroy(&x_2);
  CeedDestroy(&ceed);
  return 0;
}
