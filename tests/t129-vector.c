/// @file
/// Test filtering a vector using a threshold or tolerance
/// \test Test filtering a vector using a threshold or tolerance

//TESTARGS(name="filter none") {ceed_resource} 10 0.0
//TESTARGS(name="filter all") {ceed_resource} 10 10.0
//TESTARGS(name="between values") {ceed_resource} 10 3.5
//TESTARGS(name="equal to value") {ceed_resource} 10 7.0
//TESTARGS(name="empty vector") {ceed_resource} 0 1.0
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;
  CeedScalar tolerance = 1e-10; 

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;
  if (argc > 3) tolerance = (CeedScalar)atof(argv[3]);

  CeedVectorCreate(ceed, len, &x);
  {
    CeedScalar array[len];

    for (CeedInt i = 0; i < len; i++) array[i] = (1.0 + i)* pow(-1, i);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }
  CeedVectorFilter(x, tolerance);

  {
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
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
