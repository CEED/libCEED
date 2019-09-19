/// @file
/// Test creation and view of an element restriction
/// \test Test creation and view of an element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInt ne = 3;
  CeedInt ind[2*ne];

  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  for (CeedInt i=0; i<ne; i++) {
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed, ne, 2, ne+1, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, ind, &r);

  CeedElemRestrictionView(r, stdout);

  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
