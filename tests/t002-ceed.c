/// @file
/// Test return of a CEED object full resource name
/// \test Test return of a CEED object full resource name
#include <ceed.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  const char *backend = argv[1];
  const char *resource;

  CeedInit(backend, &ceed);

  CeedGetResource(ceed, &resource);

  const size_t resource_length               = strlen(resource);
  const bool   is_exact_match                = strcmp(resource, backend) == 0;
  const bool   is_match_with_query_arguments = !is_exact_match && memcmp(resource, backend, resource_length) == 0 && backend[resource_length] == ':';

  if (!is_exact_match && !is_match_with_query_arguments) return CeedError(ceed, 1, "Incorrect full resource name: %s != %s\n", resource, backend);

  CeedDestroy(&ceed);
  return 0;
}
