/// @file
/// Test return of CEED backend prefered memory type
/// \test Test return of CEED backend prefered memory type
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedInt type = -1;

  CeedInit(argv[1], &ceed);

  CeedGetPreferredMemType(ceed, (CeedMemType *)&type);
  if (type == -1)
    // LCOV_EXCL_START
    printf("Error getting preferred memory type. %d \n", type);
  // LCOV_EXCL_STOP

  CeedDestroy(&ceed);
  return 0;
}
