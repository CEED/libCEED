/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  CeedInt vlength = 1;
  char* spec = "elliptic";
  CeedQFunction qf;
  CeedQFunctionCreateInteriorFromGallery(ceed, vlength, spec, &qf);
  CeedDestroy(&ceed);
  return 0;
}
