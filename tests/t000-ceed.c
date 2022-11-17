/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  {
    int major, minor, patch;

    CeedGetVersion(&major, &minor, &patch, NULL);
    if (!CEED_VERSION_GE(major, minor, patch)) printf("Library version mismatch %d.%d.%d\n", major, minor, patch);
  }

  CeedInit(argv[1], &ceed);
  CeedDestroy(&ceed);

  // Test double destroy is safe
  CeedDestroy(&ceed);
  return 0;
}
