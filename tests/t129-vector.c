/// @file
/// Test filtering a vector using a threshold or tolerance
/// \test Test filtering a vector using a threshold or tolerance

//TESTARGS(name="vector of length 10") {ceed_resource} 10
//TESTARGS(name="empty vector") {ceed_resource} 0
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// Test builds the vector [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0]
// and filters it using different tolerances
static int InitVector(CeedVector x, CeedInt len) {
  if (len <= 0) return 0; // Nothing to set for an empty vector

  CeedScalar array[len];
  for (CeedInt i = 0; i < len; i++) array[i] = (1.0 + i) * pow(-1, i);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  return 0;
}

static int VerifyFilter(CeedVector x, CeedInt len, CeedScalar tolerance) {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      CeedScalar initial_value = (1.0 + i) * pow(-1, i);
      CeedScalar expected_value = (fabs(initial_value) <= tolerance) ? 0.0 : initial_value;

      if (fabs(read_array[i] - expected_value) > 1e-14) {
        // LCOV_EXCL_START
        printf("Error in filtered vector at index %" CeedInt_FMT ", computed: %f actual: %f\n", 
               i, read_array[i], expected_value);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(x, &read_array);
    return 0;
  }

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;
  CeedScalar tolerance = 1e-10; 

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);

  // Test Case 1 - tolerance between vector values
  InitVector(x, len);
  tolerance = 3.5;
  CeedVectorFilter(x, tolerance);
  VerifyFilter(x, len, tolerance);

  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }

  // Test Case 2 - tolerance equal to a vector value
  InitVector(x, len);
  tolerance = 7.0;
  CeedVectorFilter(x, tolerance);
  VerifyFilter(x, len, tolerance);

  // Test Case 3 - filter no values
  InitVector(x, len);
  tolerance = 0.0;
  CeedVectorFilter(x, tolerance);
  VerifyFilter(x, len, tolerance);

  // Test Case 4 - filter all values
  InitVector(x, len);
  tolerance = 100.0;
  CeedVectorFilter(x, tolerance);
  VerifyFilter(x, len, tolerance);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
