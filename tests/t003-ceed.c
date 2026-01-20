/// @file
/// Test viewing of a CEED object
/// \test Test viewing of a CEED object
#include <ceed.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  CeedView(ceed, stdout);

  CeedSetNumViewTabs(ceed, 1);
  CeedView(ceed, stdout);

  // Check CeedObject interface
  {
    Ceed ceed_copy = NULL;

    CeedReferenceCopy(ceed, &ceed_copy);
    CeedObjectView((CeedObject)ceed_copy, stdout);
    CeedObjectDestroy((CeedObject *)&ceed_copy);
  }

  CeedDestroy(&ceed);
  return 0;
}
