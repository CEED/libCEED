/// @file
/// Test pointwise multiplication of a pair of vectors
/// \test Test pointwise multiplication of a pair of vectors

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x, y, w;
  CeedInt           len = 10;
  const CeedScalar *read_array;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &y);
  CeedVectorCreate(ceed, len, &w);
  {
    CeedScalar array[len];

    for (CeedInt i = 0; i < len; i++) array[i] = i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
    CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }

  // Test multiplying two vectors into third
  CeedVectorPointwiseMult(w, x, y);
  CeedVectorGetArrayRead(w, CEED_MEM_HOST, &read_array);
  for (CeedInt i = 0; i < len; i++) {
    if (fabs(read_array[i] - i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in w = x .* y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], 1.0 * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(w, &read_array);

  // Test multiplying two vectors into one of the two
  CeedVectorPointwiseMult(w, w, y);
  CeedVectorGetArrayRead(w, CEED_MEM_HOST, &read_array);
  for (CeedInt i = 0; i < len; i++) {
    if (fabs(read_array[i] - i * i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in w = w .* y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], 1.0 * i * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(w, &read_array);

  // Test multiplying two vectors into one of the two
  CeedVectorPointwiseMult(w, x, w);
  CeedVectorGetArrayRead(w, CEED_MEM_HOST, &read_array);
  for (CeedInt i = 0; i < len; i++) {
    if (fabs(read_array[i] - i * i * i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in w = x .* w at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], 1.0 * i * i * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(w, &read_array);

  // Test multiplying vector by itself and putting product into self
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(y, type);
  }
  CeedVectorPointwiseMult(y, y, y);
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &read_array);
  for (CeedInt i = 0; i < len; i++) {
    if (fabs(read_array[i] - i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in y = y .* y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, read_array[i], 1.0 * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(y, &read_array);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedVectorDestroy(&w);
  CeedDestroy(&ceed);
  return 0;
}
