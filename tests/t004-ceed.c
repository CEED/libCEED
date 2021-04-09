/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  bool is_deterministic;

  CeedInit(argv[1], &ceed);
  CeedIsDeterministic(ceed, &is_deterministic);
  CeedDestroy(&ceed);
  return 0;
}
