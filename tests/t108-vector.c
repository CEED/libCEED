/// @file
/// Test vector norms
/// \test Test vector norms
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    n = 10;
  CeedScalar a[10];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i = 0; i < n; i++) a[i] = i * (i % 2 ? 1 : -1);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }

  CeedScalar norm;
  CeedVectorNorm(x, CEED_NORM_1, &norm);
  if (fabs(norm - 45.) > 100. * CEED_EPSILON) printf("Error: L1 norm %f != 45.\n", norm);

  CeedVectorNorm(x, CEED_NORM_2, &norm);
  if (fabs(norm - sqrt(285.)) > 100. * CEED_EPSILON) printf("Error: L2 norm %f != sqrt(285.)\n", norm);

  CeedVectorNorm(x, CEED_NORM_MAX, &norm);
  if (fabs(norm - 9.) > 100. * CEED_EPSILON) printf("Error: Max norm %f != 9.\n", norm);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
