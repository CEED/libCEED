/// @file
/// Test creation and view of an element restriction
/// \test Test creation and view of an element restriction
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedInt             num_elem = 3;
  CeedInt             ind[2 * num_elem];
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_elem + 1, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);

  CeedElemRestrictionView(elem_restriction, stdout);

  // Check tabs and CeedObject functionality
  {
    CeedElemRestriction elem_restriction_copy = NULL;

    CeedElemRestrictionReferenceCopy(elem_restriction, &elem_restriction_copy);
    CeedElemRestrictionSetNumViewTabs(elem_restriction_copy, 1);
    CeedObjectView((CeedObject)elem_restriction_copy, stdout);
    CeedObjectDestroy((CeedObject *)&elem_restriction_copy);
  }

  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
