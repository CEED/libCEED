/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  bool isDeterministic;

  CeedInit(argv[1], &ceed);
  CeedIsDeterministic(ceed, &isDeterministic);
  CeedDestroy(&ceed);
  return 0;
}
