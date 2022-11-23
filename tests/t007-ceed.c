/// @file
/// Test creation error message with multiple matching characters
/// \test Test creation error message with multiple matching characters
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  size_t end_index;
  for (end_index = 0; argv[1][end_index]; end_index++) {
  }
  argv[1][end_index - 1] -= 1;

  CeedInit(argv[1], &ceed);
  // LCOV_EXCL_START
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
