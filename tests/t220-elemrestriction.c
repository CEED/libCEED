/// @file
/// Test creation, use, and destruction of an oriented element restriction with unsigned application
/// \test Test creation, use, and destruction of an oriented element restriction with unsigned application
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y_oriented, y_unsigned, y_unsigned_copy;
  CeedInt             num_elem = 6, p = 2, dim = 1;
  CeedInt             ind[p * num_elem];
  bool                orient[p * num_elem];
  CeedScalar          x_array[num_elem + 1];
  CeedElemRestriction elem_restriction, elem_restriction_unsigned, elem_restriction_copy;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem + 1, &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) x_array[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, num_elem * 2, &y_oriented);
  CeedVectorCreate(ceed, num_elem * 2, &y_unsigned);
  CeedVectorCreate(ceed, num_elem * 2, &y_unsigned_copy);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0]    = i;
    ind[2 * i + 1]    = i + 1;
    orient[2 * i + 0] = (i % (2)) * -1 < 0;  // flip the dofs on element 1, 3, ...
    orient[2 * i + 1] = (i % (2)) * -1 < 0;
  }
  CeedElemRestrictionCreateOriented(ceed, num_elem, p, dim, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, orient, &elem_restriction);
  CeedElemRestrictionCreate(ceed, num_elem, p, dim, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction_unsigned);
  CeedElemRestrictionCreateUnsignedCopy(elem_restriction, &elem_restriction_copy);

  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y_oriented, CEED_REQUEST_IMMEDIATE);
  CeedElemRestrictionApply(elem_restriction_unsigned, CEED_NOTRANSPOSE, x, y_unsigned, CEED_REQUEST_IMMEDIATE);
  CeedElemRestrictionApply(elem_restriction_copy, CEED_NOTRANSPOSE, x, y_unsigned_copy, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_oriented_array, *y_unsigned_array, *y_unsigned_copy_array;

    CeedVectorGetArrayRead(y_oriented, CEED_MEM_HOST, &y_oriented_array);
    CeedVectorGetArrayRead(y_unsigned, CEED_MEM_HOST, &y_unsigned_array);
    CeedVectorGetArrayRead(y_unsigned_copy, CEED_MEM_HOST, &y_unsigned_copy_array);
    for (CeedInt i = 0; i < num_elem; i++) {
      for (CeedInt j = 0; j < p; j++) {
        CeedInt k = j + p * i;
        // unsigned application should match oriented application, but with no sign change
        if (y_oriented_array[k] * CeedIntPow(-1, i % 2) != 10 + (k + 1) / 2) {
          // LCOV_EXCL_START
          printf("Error in oriented restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_oriented_array[k]);
          // LCOV_EXCL_STOP
        }
        if (y_unsigned_array[k] != 10 + (k + 1) / 2) {
          // LCOV_EXCL_START
          printf("Error in unsigned restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_unsigned_array[k]);
          // LCOV_EXCL_STOP
        }
        if (y_unsigned_array[k] != y_unsigned_copy_array[k]) {
          // LCOV_EXCL_START
          printf("Error in copy restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_unsigned_copy_array[k]);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(y_oriented, &y_oriented_array);
    CeedVectorRestoreArrayRead(y_unsigned, &y_unsigned_array);
    CeedVectorRestoreArrayRead(y_unsigned_copy, &y_unsigned_copy_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y_oriented);
  CeedVectorDestroy(&y_unsigned);
  CeedVectorDestroy(&y_unsigned_copy);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedElemRestrictionDestroy(&elem_restriction_unsigned);
  CeedElemRestrictionDestroy(&elem_restriction_copy);
  CeedDestroy(&ceed);
  return 0;
}
