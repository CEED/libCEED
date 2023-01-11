/// @file
/// Test element restriction state counter
/// \test Test element restriction state counter
#include <ceed/backend.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedInt             num_elem = 3;
  CeedInt             ind[2 * num_elem];
  const CeedInt      *offsets;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &r);

  // Get offsets and fail to restore them
  CeedElemRestrictionGetOffsets(r, CEED_MEM_HOST, &offsets);

  CeedElemRestrictionDestroy(&r);
  // LCOV_EXCL_START
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
