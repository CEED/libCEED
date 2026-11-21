/// @file
/// Test getting the cache directory, if it can be used
/// \test Test getting the cache directory, if it can be used
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  const char *cache_dir, *base_dirs[] = {"/dev/null", "", "$(echo cache)", "`echo cache`", "\"", "a\" \"b"};

  CeedInit(argv[1], &ceed);

  // If the cache directory can be used, it is the one from CeedGetCacheDir()
  CeedTryGetCacheDir(ceed, &cache_dir);
  if (cache_dir) {
    const char *required_cache_dir;

    CeedGetCacheDir(ceed, &required_cache_dir);
    if (strcmp(cache_dir, required_cache_dir)) printf("Error: cache directories %s and %s differ\n", cache_dir, required_cache_dir);
    CeedRestoreCacheDir(ceed, &required_cache_dir);
  }
  CeedRestoreCacheDir(ceed, &cache_dir);

  // Cache base directories that cannot be created, are empty, or cannot be expanded to one word
  for (size_t i = 0; i < sizeof(base_dirs) / sizeof(base_dirs[0]); i++) {
    CeedSetCacheBaseDir(ceed, base_dirs[i]);
    CeedTryGetCacheDir(ceed, &cache_dir);
    if (cache_dir) printf("Error: cache directory %s for cache base directory %s\n", cache_dir, base_dirs[i]);
    CeedRestoreCacheDir(ceed, &cache_dir);
  }
  CeedSetCacheBaseDir(ceed, NULL);
  CeedTryGetCacheDir(ceed, &cache_dir);
  if (cache_dir) printf("Error: cache directory %s without cache base directory\n", cache_dir);
  CeedRestoreCacheDir(ceed, &cache_dir);

  CeedDestroy(&ceed);
  return 0;
}
