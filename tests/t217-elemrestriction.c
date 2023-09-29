/// @file
/// Test creation, use, and destruction of an oriented element restriction
/// \test Test creation, use, and destruction of an oriented element restriction
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 6, elem_size = 2;
  CeedInt             ind[elem_size * num_elem];
  bool                orients[elem_size * num_elem];
  CeedScalar          x_array[num_elem + 1];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem + 1, &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) x_array[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, num_elem * elem_size, &y);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0]     = i;
    ind[2 * i + 1]     = i + 1;
    orients[2 * i + 0] = (i % 2) > 0;  // flip the dofs on element 1, 3, ...
    orients[2 * i + 1] = (i % 2) > 0;
  }
  CeedElemRestrictionCreateOriented(ceed, num_elem, elem_size, 1, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, orients, &elem_restriction);

  // NoTranspose
  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    for (CeedInt i = 0; i < num_elem; i++) {
      for (CeedInt j = 0; j < elem_size; j++) {
        CeedInt k = j + elem_size * i;
        if (y_array[k] * CeedIntPow(-1, i % 2) != 10 + (k + 1) / 2) {
          // LCOV_EXCL_START
          printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_array[k]);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(y, &y_array);
  }

  // Transpose
  CeedVectorSetValue(x, 0);
  CeedElemRestrictionApply(elem_restriction, CEED_TRANSPOSE, y, x, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *x_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array);
    for (CeedInt i = 0; i < num_elem + 1; i++) {
      if (x_array[i] != (10 + i) * (i > 0 && i < num_elem ? 2.0 : 1.0))
        // LCOV_EXCL_START
        printf("Error in restricted array x[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)x_array[i]);
      // LCOV_EXCL_STOP
    }
    CeedVectorRestoreArrayRead(x, &x_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
