/// @file
/// Test clearing work vectors
/// \test Test clearing work vectors

#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>

static CeedScalar expected_usage(CeedSize length) { return length * sizeof(CeedScalar) * 1e-6; }

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x, y, z;
  CeedScalar usage_mb;

  CeedInit(argv[1], &ceed);

  // Add work vectors of different lengths
  CeedGetWorkVector(ceed, 10, &x);
  CeedGetWorkVector(ceed, 20, &y);
  CeedGetWorkVector(ceed, 30, &z);

  // Check memory usage, should be 60 * sizeof(CeedScalar)
  CeedGetWorkVectorMemoryUsage(ceed, &usage_mb);
  if (fabs(usage_mb - expected_usage(60)) > 100. * CEED_EPSILON) printf("Wrong usage: %0.8g MB != %0.8g MB\n", usage_mb, expected_usage(60));

  // Restore x and z
  CeedRestoreWorkVector(ceed, &x);
  CeedRestoreWorkVector(ceed, &z);

  // Clear work vectors with length < 30. This should:
  //  - Remove x
  //  - Leave y, since it is still in use
  //  - Leave z, since it is length 30
  CeedClearWorkVectors(ceed, 30);
  CeedGetWorkVectorMemoryUsage(ceed, &usage_mb);
  if (fabs(usage_mb - expected_usage(50)) > 100. * CEED_EPSILON) printf("Wrong usage: %0.8g MB != %0.8g MB\n", usage_mb, expected_usage(50));

  // Clear work vectors with length < 31. This should:
  //  - Leave y, since it is still in use
  //  - Remove z
  CeedClearWorkVectors(ceed, 31);
  CeedGetWorkVectorMemoryUsage(ceed, &usage_mb);
  if (fabs(usage_mb - expected_usage(20)) > 100. * CEED_EPSILON) printf("Wrong usage: %0.8g MB != %0.8g MB\n", usage_mb, expected_usage(20));

  // Restore y
  CeedRestoreWorkVector(ceed, &y);

  // Make sure we can still get back y without allocating a new work vector
  CeedGetWorkVector(ceed, 20, &y);
  CeedGetWorkVectorMemoryUsage(ceed, &usage_mb);
  if (fabs(usage_mb - expected_usage(20)) > 100. * CEED_EPSILON) printf("Wrong usage: %0.8g MB != %0.8g MB\n", usage_mb, expected_usage(20));
  CeedRestoreWorkVector(ceed, &y);

  CeedDestroy(&ceed);
  return 0;
}
