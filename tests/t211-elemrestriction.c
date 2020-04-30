/// @file
/// Test creation and view of a strided element restriction
/// \test Test creation and view of a strided element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInt ne = 3;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  CeedInt strides[3] = {1, 2, 2};
  CeedElemRestrictionCreateStrided(ceed, ne, 2, 1, ne+1, strides, &r);

  CeedElemRestrictionView(r, stdout);

  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
