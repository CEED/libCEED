/// @file
/// Test return of a CEED object full resource name
/// \test Test return of a CEED object full resource name
#include <string.h>
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  const char *resource;

  CeedInit(argv[1], &ceed);

  CeedGetResource(ceed, &resource);
  if (strcmp(resource, argv[1]))
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Incorrect full resource name: %s != %s",
                     resource, argv[1]);
  // LCOV_EXCL_STOP

  CeedDestroy(&ceed);
  return 0;
}
