/// @file
/// Test creation help message
/// \test Test creation help message
#include <ceed.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed ceed;

  char help_resource[256];
  sprintf(help_resource, "help:%s", argv[1]);

  CeedInit(help_resource, &ceed);
  CeedDestroy(&ceed);

  return 0;
}
