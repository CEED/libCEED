/// @file
/// Test creation, use, and destruction of a curl-conforming oriented element restriction with and without unsigned application
/// \test Test creation, use, and destruction of a curl-conforming oriented element restriction with and without unsigned application
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y, y_unsigned;
  CeedInt             num_elem = 6, elem_size = 4;
  CeedInt             ind[elem_size * num_elem];
  CeedInt8            curl_orients[3 * elem_size * num_elem];
  CeedScalar          x_array[3 * num_elem + 1];
  CeedElemRestriction elem_restriction, elem_restriction_unsigned;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 3 * num_elem + 1, &x);
  for (CeedInt i = 0; i < 3 * num_elem + 1; i++) x_array[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, num_elem * elem_size, &y);
  CeedVectorCreate(ceed, num_elem * elem_size, &y_unsigned);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[4 * i + 0] = 3 * i + 0;
    ind[4 * i + 1] = 3 * i + 1;
    ind[4 * i + 2] = 3 * i + 2;
    ind[4 * i + 3] = 3 * i + 3;
    if (i % 2 > 0) {
      // T = [ 1  0  0  0]
      //     [ 0  1  0  0]
      //     [ 0  0  0 -1]
      //     [ 0  0 -1  0]
      curl_orients[3 * 4 * i + 0]  = 0;
      curl_orients[3 * 4 * i + 1]  = 1;
      curl_orients[3 * 4 * i + 2]  = 0;
      curl_orients[3 * 4 * i + 3]  = 0;
      curl_orients[3 * 4 * i + 4]  = 1;
      curl_orients[3 * 4 * i + 5]  = 0;
      curl_orients[3 * 4 * i + 6]  = 0;
      curl_orients[3 * 4 * i + 7]  = 0;
      curl_orients[3 * 4 * i + 8]  = -1;
      curl_orients[3 * 4 * i + 9]  = -1;
      curl_orients[3 * 4 * i + 10] = 0;
      curl_orients[3 * 4 * i + 11] = 0;
    } else {
      // T = I
      curl_orients[3 * 4 * i + 0]  = 0;
      curl_orients[3 * 4 * i + 1]  = 1;
      curl_orients[3 * 4 * i + 2]  = 0;
      curl_orients[3 * 4 * i + 3]  = 0;
      curl_orients[3 * 4 * i + 4]  = 1;
      curl_orients[3 * 4 * i + 5]  = 0;
      curl_orients[3 * 4 * i + 6]  = 0;
      curl_orients[3 * 4 * i + 7]  = 1;
      curl_orients[3 * 4 * i + 8]  = 0;
      curl_orients[3 * 4 * i + 9]  = 0;
      curl_orients[3 * 4 * i + 10] = 1;
      curl_orients[3 * 4 * i + 11] = 0;
    }
  }
  CeedElemRestrictionCreateCurlOriented(ceed, num_elem, elem_size, 1, 1, 3 * num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, curl_orients,
                                        &elem_restriction);
  CeedElemRestrictionCreateUnsignedCopy(elem_restriction, &elem_restriction_unsigned);

  // NoTranspose
  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  CeedElemRestrictionApply(elem_restriction_unsigned, CEED_NOTRANSPOSE, x, y_unsigned, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_array, *y_unsigned_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    CeedVectorGetArrayRead(y_unsigned, CEED_MEM_HOST, &y_unsigned_array);
    for (CeedInt i = 0; i < num_elem; i++) {
      for (CeedInt j = 0; j < elem_size; j++) {
        CeedInt k = j + elem_size * i;
        if (i % 2 > 0 && j >= 2) {
          if (j == 2 && 10 + 3 * i + j + 1 != -y_array[k]) {
            // LCOV_EXCL_START
            printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_array[k]);
            // LCOV_EXCL_STOP
          } else if (j == 3 && 10 + 3 * i + j - 1 != -y_array[k]) {
            // LCOV_EXCL_START
            printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_array[k]);
            // LCOV_EXCL_STOP
          }
          if (j == 2 && 10 + 3 * i + j + 1 != y_unsigned_array[k]) {
            // LCOV_EXCL_START
            printf("Error in unsigned restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_unsigned_array[k]);
            // LCOV_EXCL_STOP
          } else if (j == 3 && 10 + 3 * i + j - 1 != y_unsigned_array[k]) {
            // LCOV_EXCL_START
            printf("Error in unsigned restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_unsigned_array[k]);
            // LCOV_EXCL_STOP
          }
        } else {
          if (10 + 3 * i + j != y_array[k] || 10 + 3 * i + j != y_unsigned_array[k]) {
            // LCOV_EXCL_START
            printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_array[k]);
            // LCOV_EXCL_STOP
          }
        }
      }
    }
    CeedVectorRestoreArrayRead(y, &y_array);
    CeedVectorRestoreArrayRead(y_unsigned, &y_unsigned_array);
  }

  // Transpose
  CeedVectorSetValue(x, 0);
  CeedElemRestrictionApply(elem_restriction, CEED_TRANSPOSE, y, x, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *x_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array);
    for (CeedInt i = 0; i < 3 * num_elem + 1; i++) {
      if (x_array[i] != (10 + i) * (i > 0 && i < 3 * num_elem && i % 3 == 0 ? 2.0 : 1.0)) {
        // LCOV_EXCL_START
        printf("Error in restricted array x[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)x_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(x, &x_array);
  }

  // Transpose unsigned
  CeedVectorSetValue(x, 0);
  CeedElemRestrictionApply(elem_restriction_unsigned, CEED_TRANSPOSE, y_unsigned, x, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *x_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array);
    for (CeedInt i = 0; i < 3 * num_elem + 1; i++) {
      if (x_array[i] != (10 + i) * (i > 0 && i < 3 * num_elem && i % 3 == 0 ? 2.0 : 1.0)) {
        // LCOV_EXCL_START
        printf("Error in restricted array x[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)x_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(x, &x_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedVectorDestroy(&y_unsigned);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedElemRestrictionDestroy(&elem_restriction_unsigned);
  CeedDestroy(&ceed);
  return 0;
}
