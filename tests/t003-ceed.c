/// @file
/// Test viewing of a CEED object full
/// \test Test viewing of a CEED object
#include <string.h>
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  CeedView(ceed, stdout);

  CeedDestroy(&ceed);
  return 0;
}
