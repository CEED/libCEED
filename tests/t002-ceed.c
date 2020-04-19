/// @file
/// Test return of a CEED object full resource name
/// \test Test return of a CEED object full resource name
#include <string.h>
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  const char *backend = argv[1];
  const char *resource;

  CeedInit(backend, &ceed);

  CeedGetResource(ceed, &resource);

  const size_t resourceLength = strlen(resource);
  const bool isExactMatch = strcmp(resource, backend) == 0;
  const bool isMatchWithQueryArguments = (
      !isExactMatch
      && memcmp(resource, backend, resourceLength) == 0
      && backend[resourceLength] == ':'
                                         );

  if (!isExactMatch && !isMatchWithQueryArguments) {
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Incorrect full resource name: %s != %s",
                     resource, backend);
    // LCOV_EXCL_STOP
  }

  CeedDestroy(&ceed);
  return 0;
}
