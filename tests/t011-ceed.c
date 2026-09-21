/// @file
/// Test getting/setting environment variables
/// \test Test viewing of a CEED object
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed ceed, ceed_delegate, ceed_test = NULL;

  CeedInit(argv[1], &ceed);
  CeedGetDelegate(ceed, &ceed_delegate);

#define CEED_ENV_FLAG(suffix, default, ...)                                                                                           \
  {                                                                                                                                   \
    bool value, new_value;                                                                                                            \
                                                                                                                                      \
    CeedGet##suffix(ceed_test, &value);                                                                                               \
    CeedSet##suffix(ceed_test, !value);                                                                                               \
    CeedGet##suffix(ceed_test, &new_value);                                                                                           \
    if (new_value != !value) {                                                                                                        \
      printf("Error: failed to set flag %s (set %s, actual %s)\n", #suffix, !value ? "true" : "false", new_value ? "true" : "false"); \
    }                                                                                                                                 \
    CeedSet##suffix(ceed_test, value);                                                                                                \
  }
#define CEED_ENV_ONLY_FLAG(suffix, default, ...)
#define CEED_ENV_STRING(suffix, default, ...)                                                                       \
  {                                                                                                                 \
    const char *env_value, *value, *new_value = "test value";                                                       \
    char       *original_value;                                                                                     \
                                                                                                                    \
    CeedGet##suffix(ceed_test, &value);                                                                             \
    if (value) {                                                                                                    \
      size_t len = strlen(value);                                                                                   \
                                                                                                                    \
      original_value = calloc(len + 1, sizeof(char));                                                               \
      memcpy(original_value, value, len);                                                                           \
    } else {                                                                                                        \
      original_value = NULL;                                                                                        \
    }                                                                                                               \
    CeedSet##suffix(ceed_test, new_value);                                                                          \
    CeedGet##suffix(ceed_test, &value);                                                                             \
    if (!value || !strstr(new_value, value)) {                                                                      \
      printf("Error: failed to set string %s (set %s, actual %s)\n", #suffix, new_value, value ? value : "(null)"); \
    }                                                                                                               \
                                                                                                                    \
    CeedSet##suffix(ceed_test, NULL);                                                                               \
    CeedGet##suffix(ceed_test, &value);                                                                             \
    if (value) {                                                                                                    \
      printf("Error: failed to set string %s (set (null), actual %s)\n", #suffix, value);                           \
    }                                                                                                               \
    CeedGetEnv##suffix(&env_value);                                                                                 \
    if (((!env_value && !original_value) && (env_value != original_value)) ||                                       \
        ((env_value && original_value) && !strstr(env_value, original_value))) {                                    \
      printf("Error: original value doesn't match environment for string %s (orig %s, env %s)\n", #suffix,          \
             original_value ? original_value : "(null)", env_value ? env_value : "(null)");                         \
    }                                                                                                               \
    CeedSet##suffix(ceed_test, env_value);                                                                          \
    if (original_value) free(original_value);                                                                       \
  }
#define CEED_ENV_ONLY_STRING(suffix, default, ...)

  CeedReferenceCopy(ceed, &ceed_test);
#include <ceed/ceed-env-list.h>
  if (ceed_delegate) {
    CeedReferenceCopy(ceed_delegate, &ceed_test);
#include <ceed/ceed-env-list.h>
  }
  CeedDestroy(&ceed_test);
#undef CEED_ENV_FLAG
#undef CEED_ENV_ONLY_FLAG
#undef CEED_ENV_STRING
#undef CEED_ENV_ONLY_STRING

  CeedDestroy(&ceed_delegate);
  CeedDestroy(&ceed);
  return 0;
}
