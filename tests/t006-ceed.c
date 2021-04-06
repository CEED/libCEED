/// @file
/// Test creation error message with only 1 matching character
/// \test Test creation error message with only 1 matching character
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  argv[1][1] = 'd';
  CeedInit(argv[1], &ceed);
  CeedDestroy(&ceed);

  return 0;
}
