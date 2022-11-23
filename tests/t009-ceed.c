/// @file
/// Test creation, copying, and destruction of a CEED object
/// \test Test creation, copying, and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed, ceed_2;

  CeedInit(argv[1], &ceed);
  CeedInit("/cpu/self/ref/serial", &ceed_2);

  CeedReferenceCopy(ceed, &ceed_2);  // This destroys the previous ceed_2
  if (ceed != ceed_2) printf("Error copying Ceed reference\n");

  CeedDestroy(&ceed);

  CeedMemType type;
  CeedGetPreferredMemType(ceed_2, &type);  // Second reference still valid

  CeedDestroy(&ceed_2);  // Both references should be destroyed
  return 0;
}
