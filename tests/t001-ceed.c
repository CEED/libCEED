/// @file
/// Test return of CEED backend preferred memory type
/// \test Test return of CEED backend preferred memory type
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed    ceed;
  CeedInt type = -1;

  CeedInit(argv[1], &ceed);

  CeedGetPreferredMemType(ceed, (CeedMemType *)&type);
  if (type == -1) printf("Error getting preferred memory type. %" CeedInt_FMT "\n", type);

  CeedDestroy(&ceed);
  return 0;
}
