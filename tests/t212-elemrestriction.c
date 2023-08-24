/// @file
/// Test creation and view of a blocked strided element restriction
/// \test Test creation and view of a blocked strided element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInt             num_elem = 3;
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedInt strides[3] = {1, 2, 2};
  CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, 2, 2, 1, num_elem * 2, strides, &elem_restriction);

  CeedElemRestrictionView(elem_restriction, stdout);

  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
