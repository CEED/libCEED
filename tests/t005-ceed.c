/// @file
/// Test error storage for a CEED object
/// \test Test error storage for a CEED object
#include <ceed.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  // Check for standard message with default handler
  const char *err_msg = NULL;
  CeedGetErrorMessage(ceed, &err_msg);
  if (strcmp(err_msg, "No error message stored"))
    // LCOV_EXCL_START
    printf("Unexpected error message received: \"%s\"\n", err_msg);
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
  CeedGetErrorMessage(ceed, &err_msg);
  if (!err_msg || !strcmp(err_msg, "No error message stored"))
    // LCOV_EXCL_START
    printf("Unexpected error message received: \"%s\"\n", err_msg);
  // LCOV_EXCL_STOP
  CeedResetErrorMessage(ceed, &err_msg);

  // Check error message reset
  CeedGetErrorMessage(ceed, &err_msg);
  if (strcmp(err_msg, "No error message stored"))
    // LCOV_EXCL_START
    printf("Unexpected error message received: \"%s\"\n", err_msg);
  // LCOV_EXCL_STOP

  // Cleanup
  CeedVectorRestoreArray(vec, &array);
  CeedVectorDestroy(&vec);
  CeedDestroy(&ceed);
  return 0;
}
