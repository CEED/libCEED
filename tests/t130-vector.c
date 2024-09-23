/// @file
/// Test getting and restoring work vectors
/// \test Test getting and restoring work vectors

#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  // Check for getting the same work vector back
  {
    CeedVector x, y;

    CeedGetWorkVector(ceed, 20, &x);
    // Do not do this!
    CeedVector x_copy = x;

    CeedRestoreWorkVector(ceed, &x);
    CeedGetWorkVector(ceed, 20, &y);
    if (y != x_copy) printf("failed to return same work vector");
    CeedRestoreWorkVector(ceed, &y);
  }

  // Check for getting a new work vector back
  {
    CeedVector x, y;

    CeedGetWorkVector(ceed, 20, &x);
    // Do not do this!
    CeedVector x_copy = x;

    CeedRestoreWorkVector(ceed, &x);
    CeedGetWorkVector(ceed, 30, &y);
    if (y == x_copy) printf("failed to return new work vector");
    CeedRestoreWorkVector(ceed, &y);
  }

  CeedDestroy(&ceed);
  return 0;
}
