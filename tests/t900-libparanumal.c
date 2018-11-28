/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
//  CeedCreateQFunctionFromGallery();
  CeedDestroy(&ceed);
  return 0;
}
