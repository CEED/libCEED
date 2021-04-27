/// @file
/// Test creation, use, and destruction of a blocked element restriction
/// \test Test creation, use, and destruction of a blocked element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt num_elem = 8;
  CeedInt blk_size = 5;
  CeedInt elem_size = 2;
  CeedInt ind[elem_size*num_elem];
  CeedScalar a[num_elem+1];
  CeedElemRestriction r;
  CeedScalar *y_array;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_elem+1, &x);
  for (CeedInt i=0; i<num_elem+1; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<num_elem; i++) {
    for (CeedInt k=0; k<elem_size; k++) {
      ind[elem_size*i+k] = i+k;
    }
  }

  CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, blk_size, 1, 1,
                                   num_elem+1,
                                   CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);

  CeedVectorCreate(ceed, blk_size*elem_size, &y);
  CeedVectorSetValue(y, 0); // Allocates array

  // NoTranspose
  CeedElemRestrictionApplyBlock(r, 1, CEED_NOTRANSPOSE, x, y,
                                CEED_REQUEST_IMMEDIATE);

  // Zero padded entries
  CeedVectorGetArray(y, CEED_MEM_HOST, &y_array);
  for (CeedInt i = (elem_size*num_elem - blk_size*elem_size);
       i < blk_size*elem_size;
       ++i) {
    y_array[i] = 0;
  }
  CeedVectorRestoreArray(y, &y_array);
  CeedVectorView(y, "%12.8f", stdout);

  // Transpose
  CeedVectorGetArray(x, CEED_MEM_HOST, (CeedScalar **)&a);
  for (CeedInt i=0; i<num_elem+1; i++)
    a[i] = 0;
  CeedVectorRestoreArray(x, (CeedScalar **)&a);
  CeedElemRestrictionApplyBlock(r, 1, CEED_TRANSPOSE, y, x,
                                CEED_REQUEST_IMMEDIATE);
  CeedVectorView(x, "%12.8f", stdout);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
