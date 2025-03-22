/// @file
/// Test git version and build configuration
/// \test Test git version and build configuration
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  const char *git_version, *build_config;
  CeedGetGitVersion(&git_version);
  CeedGetBuildConfiguration(&build_config);
  // printf("Git: %s\n", git_version);
  // puts(build_config);
  return 0;
}
