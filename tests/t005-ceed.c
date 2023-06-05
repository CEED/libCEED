/// @file
/// Test error storage for a CEED object
/// \test Test error storage for a CEED object
#include <ceed.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  vec;
  CeedScalar *array;
  const char *err_msg = NULL;

  CeedInit(argv[1], &ceed);

  // Check for standard message with default handler
  CeedGetErrorMessage(ceed, &err_msg);
  if (strcmp(err_msg, "No error message stored")) printf("Unexpected error message received: \"%s\"\n", err_msg);

  // Set error handler to store error message
  CeedSetErrorHandler(ceed, CeedErrorStore);

  // Generate error
  CeedVectorCreate(ceed, 10, &vec);
  CeedVectorGetArray(vec, CEED_MEM_HOST, &array);
  CeedVectorGetArray(vec, CEED_MEM_HOST, &array);

  // Check error message
  CeedGetErrorMessage(ceed, &err_msg);
  if (!err_msg || !strcmp(err_msg, "No error message stored\n")) printf("Unexpected error message received: \"%s\"\n", err_msg);
  CeedResetErrorMessage(ceed, &err_msg);

  // Check error message reset
  CeedGetErrorMessage(ceed, &err_msg);
  if (strcmp(err_msg, "No error message stored")) printf("Unexpected error message received: \"%s\"\n", err_msg);

  // Cleanup
  CeedVectorRestoreArray(vec, &array);
  CeedVectorDestroy(&vec);
  CeedDestroy(&ceed);
  return 0;
}
