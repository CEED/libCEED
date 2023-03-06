/// @file
/// Test creation, use, and destruction of an element restriction
/// \test Test creation, use, and destruction of an element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 3;
  CeedInt             ind[2 * num_elem];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem + 1, &x);
  {
    CeedScalar array[num_elem + 1];

    for (CeedInt i = 0; i < num_elem + 1; i++) array[i] = 10 + i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }
  CeedVectorCreate(ceed, num_elem * 2, &y);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);
  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < num_elem * 2; i++) {
      if (10 + (i + 1) / 2 != read_array[i]) printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)read_array[i]);
    }
    CeedVectorRestoreArrayRead(y, &read_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
