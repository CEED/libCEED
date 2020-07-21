/// @file
/// Test error storage for a CEED object
/// \test Test error storage for a CEED object
#include <ceed.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  // Check for standard message with default handler
  const char *errmsg = NULL;
  CeedGetErrorMessage(ceed, &errmsg);
  if (strcmp(errmsg, "No error message stored"))
    // LCOV_EXCL_START
    printf("Unexpected error message received: \"%s\"\n", errmsg);
  // LCOV_EXCL_STOP

  // Set error handler to store error message
  CeedSetErrorHandler(ceed, CeedErrorStore);

  // Generate error
  CeedVector vec;
  CeedScalar *array;
  CeedVectorCreate(ceed, 10, &vec);
  CeedVectorGetArray(vec, CEED_MEM_HOST, &array);
  CeedVectorGetArray(vec, CEED_MEM_HOST, &array);

  // Check error message
  CeedGetErrorMessage(ceed, &errmsg);
  if (!errmsg || !strcmp(errmsg, "No error message stored"))
    // LCOV_EXCL_START
    printf("Unexpected error message received: \"%s\"\n", errmsg);
  // LCOV_EXCL_STOP
  CeedResetErrorMessage(ceed, &errmsg);

  // Check error message reset
  CeedGetErrorMessage(ceed, &errmsg);
  if (strcmp(errmsg, "No error message stored"))
    // LCOV_EXCL_START
    printf("Unexpected error message received: \"%s\"\n", errmsg);
  // LCOV_EXCL_STOP

  // Cleanup
  CeedVectorRestoreArray(vec, &array);
  CeedVectorDestroy(&vec);
  CeedDestroy(&ceed);
  return 0;
}
