/// @file
/// Test creation, use, and destruction of an element restriction oriented
/// \test Test creation, use, and destruction of an element restriction oriented
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 6, p = 2, dim = 1;
  CeedInt             ind[p * num_elem];
  bool                orient[p * num_elem];
  CeedScalar          x_array[num_elem + 1];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem + 1, &x);
  for (CeedInt i = 0; i < num_elem + 1; i++) x_array[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, x_array);
  CeedVectorCreate(ceed, num_elem * 2, &y);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
    // flip the dofs on element 1,3,...
    orient[2 * i + 0] = (i % (2)) * -1 < 0;
    orient[2 * i + 1] = (i % (2)) * -1 < 0;
  }
  CeedElemRestrictionCreateOriented(ceed, num_elem, p, dim, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, orient, &elem_restriction);

  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *y_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &y_array);
    for (CeedInt i = 0; i < num_elem; i++) {
      for (CeedInt j = 0; j < p; j++) {
        CeedInt k = j + p * i;
        if (y_array[k] * CeedIntPow(-1, i % 2) != 10 + (k + 1) / 2) {
          // LCOV_EXCL_START
          printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", k, (CeedScalar)y_array[k]);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(y, &y_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
