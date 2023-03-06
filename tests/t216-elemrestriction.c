/// @file
/// Test creation, copying, and destruction of an element restriction
/// \test Test creation, copying, and destruction of an element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedInt             num_elem = 3, comp_stride = 1;
  CeedInt             ind[2 * num_elem];
  CeedElemRestriction elem_restriction, elem_restriction_2;

  CeedInit(argv[1], &ceed);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, comp_stride, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, comp_stride + 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction_2);

  CeedElemRestrictionReferenceCopy(elem_restriction, &elem_restriction_2);  // This destroys the previous r_2
  CeedElemRestrictionDestroy(&elem_restriction);

  CeedInt comp_stride_2;
  CeedElemRestrictionGetCompStride(elem_restriction_2, &comp_stride_2);
  if (comp_stride_2 != comp_stride) printf("Error copying CeedElemRestriction reference\n");

  CeedElemRestrictionDestroy(&elem_restriction_2);
  CeedDestroy(&ceed);
  return 0;
}
