/// @file
/// Test vector norms
/// \test Test vector norms

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <math.h>
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
  for (CeedInt i = 0; i < len; i++) array[i] = i * (i % 2 ? 1 : -1);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, array);
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }

  CeedScalar norm;
  CeedVectorNorm(x, CEED_NORM_1, &norm);
  if (len > 0 && fabs(norm - 45.) > 100. * CEED_EPSILON) printf("Error: L1 norm %f != 45.\n", norm);
  else if (len == 0 && fabs(norm) > CEED_EPSILON) printf("Error: L1 norm %f != 0.\n", norm);

  CeedVectorNorm(x, CEED_NORM_2, &norm);
  if (len > 0 && fabs(norm - sqrt(285.)) > 100. * CEED_EPSILON) printf("Error: L2 norm %f != sqrt(285.)\n", norm);
  else if (len == 0 && fabs(norm) > CEED_EPSILON) printf("Error: L2 norm %f != 0.\n", norm);

  CeedVectorNorm(x, CEED_NORM_MAX, &norm);
  if (len > 0 && fabs(norm - 9.) > 100. * CEED_EPSILON) printf("Error: Max norm %f != 9.\n", norm);
  else if (len == 0 && fabs(norm) > CEED_EPSILON) printf("Error: Max norm %f != 0.\n", norm);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
