/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed ceed;

  {
    int major, minor, patch;

    CeedGetVersion(&major, &minor, &patch, NULL);
    if (!CEED_VERSION_GE(major, minor, patch)) printf("Library version mismatch %d.%d.%d\n", major, minor, patch);
  }

  CeedRegisterAll();  // Note, normally should not be called by users
  {
    size_t       num_backends = 0;
    char *const *resources;
    CeedInt     *priorities;

    CeedRegistryGetList(&num_backends, &resources, &priorities);
    if (num_backends == 0) printf("Error retrieving backend list");
    for (size_t i = 0; i < num_backends; i++) {
      if (!resources[i]) printf("Error retrieving resource name %ld\n", i);
      if (priorities[i] < 1) printf("Error retrieving resource priority %ld\n", i);
    }
    free((void *)resources);
    free(priorities);
  }

  CeedInit(argv[1], &ceed);
  CeedDestroy(&ceed);

  // Test double destroy is safe
  CeedDestroy(&ceed);
  return 0;
}
