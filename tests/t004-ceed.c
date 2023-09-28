/// @file
/// Test checking deterministic status of a CEED object
/// \test Test checking deterministic status a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  bool is_deterministic;

  CeedInit(argv[1], &ceed);
  CeedIsDeterministic(ceed, &is_deterministic);
  CeedDestroy(&ceed);
  return 0;
}
