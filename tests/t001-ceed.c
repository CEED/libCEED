/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>
#include <limits.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedMemType type = INT_MAX;

  CeedInit(argv[1], &ceed);
  CeedGetPreferredMemType(ceed, &type);

  if (type == INT_MAX)
    printf("Error getting preferred memory type. %d \n",type);

  CeedDestroy(&ceed);
  return 0;
}
