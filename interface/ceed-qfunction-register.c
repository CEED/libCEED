#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdlib.h>

static bool register_all_called;

#define MACRO(name) CEED_INTERN int name(void);
#include "../gallery/ceed-gallery-list.h"
#undef MACRO

/**
  @brief Register the gallery of preconfigured QFunctions.

  This is called automatically by CeedQFunctionCreateInteriorByName() and thus normally need not be called by users.
  Users can call CeedQFunctionRegister() to register additional backends.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedQFunctionRegister()

  @ref User
**/
int CeedQFunctionRegisterAll() {
  if (register_all_called) return 0;

  if (getenv("CEED_DEBUG")) {
    fflush(stdout);
    fprintf(stdout, "\033[38;5;%dm", 1);
    fprintf(stdout, "---------- Registering Gallery QFunctions ----------\n");
    fprintf(stdout, "\033[m");
    fprintf(stdout, "\n");
    fflush(stdout);
  }

  register_all_called = true;

#define MACRO(name) CeedChk(name());
#include "../gallery/ceed-gallery-list.h"
#undef MACRO

  if (getenv("CEED_DEBUG")) {
    fprintf(stdout, "\n");
    fflush(stdout);
  }

  return CEED_ERROR_SUCCESS;
}
