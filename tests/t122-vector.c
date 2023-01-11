/// @file
/// Test pointwise muliplication of a pair of vectors
/// \test Test pointwise muliplication of a pair of vectors
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x, y, w;
  CeedInt           n;
  CeedScalar        a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorCreate(ceed, n, &y);
  CeedVectorCreate(ceed, n, &w);
  for (CeedInt i = 0; i < n; i++) a[i] = i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);
  CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, a);

  // Test multiplying two vectors into third
  CeedVectorPointwiseMult(w, x, y);
  CeedVectorGetArrayRead(w, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (fabs(b[i] - i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in w = x .* y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, b[i], 1.0 * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(w, &b);

  // Test multiplying two vectors into one of the two
  CeedVectorPointwiseMult(w, w, y);
  CeedVectorGetArrayRead(w, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (fabs(b[i] - i * i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in w = w .* y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, b[i], 1.0 * i * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(w, &b);

  // Test multiplying two vectors into one of the two
  CeedVectorPointwiseMult(w, x, w);
  CeedVectorGetArrayRead(w, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (fabs(b[i] - i * i * i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in w = x .* w at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, b[i], 1.0 * i * i * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(w, &b);

  // Test multiplying vector by itself and putting product into self
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(y, type);
  }
  CeedVectorPointwiseMult(y, y, y);
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &b);
  for (CeedInt i = 0; i < n; i++) {
    if (fabs(b[i] - i * i) > 1e-14) {
      // LCOV_EXCL_START
      printf("Error in y = y .* y at index %" CeedInt_FMT ", computed: %f actual: %f\n", i, b[i], 1.0 * i * i);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(y, &b);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedVectorDestroy(&w);
  CeedDestroy(&ceed);
  return 0;
}
