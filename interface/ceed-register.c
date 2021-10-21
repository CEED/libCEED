#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed-impl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static bool register_all_called;

#define MACRO(name,...) CEED_INTERN int name(void);
#include "../backends/ceed-backend-list.h"
#undef MACRO

/**
  @brief Register all preconfigured backends.

  This is called automatically by CeedInit() and thus normally need not be called by users.
  Users can call CeedRegister() to register additional backends.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedRegister()

  @ref User
**/
int CeedRegisterAll() {
  if (register_all_called) return 0;

  if (getenv("CEED_DEBUG")) fprintf(stderr,
                                      "---------- Registering Backends ----------\n");

  register_all_called = true;

#define MACRO(name,...) CeedChk(name());
#include "../backends/ceed-backend-list.h"
#undef MACRO
  return CEED_ERROR_SUCCESS;
}
