#define _POSIX_C_SOURCE 200112
#include <feme-impl.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
  char prefix[FEME_MAX_RESOURCE_LEN];
  int (*init)(const char *resource, Feme f);
} backends[32];
static size_t num_backends;

int FemeError(Feme feme, int ecode, const char *format, ...) {
  va_list args;
  va_start(args, format);
  if (feme) return feme->Error(feme, ecode, format, args);
  return FemeErrorAbort(feme, ecode, format, args);
}

int FemeErrorReturn(Feme feme, int ecode, const char *format, va_list args) {
  return ecode;
}

int FemeErrorAbort(Feme feme, int ecode, const char *format, va_list args) {
  vfprintf(stderr, format, args);
  exit(ecode);
  return ecode;
}

int FemeRegister(const char *prefix, int (*init)(const char *resource, Feme f)) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0])) {
    return FemeError(NULL, 1, "Too many backends");
  }
  strncpy(backends[num_backends].prefix, prefix, FEME_MAX_RESOURCE_LEN);
  backends[num_backends].init = init;
  num_backends++;
  return 0;
}

int FemeMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void**)p, FEME_ALIGN, n*unit);
  if (ierr) return FemeError(NULL, ierr, "posix_memalign failed to allocate %zd members of size %zd\n", n, unit);
  return 0;
}

int FemeCallocArray(size_t n, size_t unit, void *p) {
  *(void**)p = calloc(n, unit);
  if (n && unit && !*(void**)p) return FemeError(NULL, 1, "calloc failed to allocate %zd members of size %zd\n", n, unit);
  return 0;
}

// Takes void* to avoid needing a cast, but is the address of the pointer.
int FemeFree(void *p) {
  free(*(void**)p);
  *(void**)p = NULL;
  return 0;
}

int FemeInit(const char *resource, Feme *feme) {
  int ierr;
  size_t matchlen = 0, matchidx;
  for (size_t i=0; i<num_backends; i++) {
    size_t n;
    const char *prefix = backends[i].prefix;
    for (n = 0; prefix[n] && prefix[n] == resource[n]; n++) {}
    if (n > matchlen) {
      matchlen = n;
      matchidx = i;
    }
  }
  if (!matchlen) return FemeError(NULL, 1, "No suitable backend");
  ierr = FemeCalloc(1,feme);FemeChk(ierr);
  (*feme)->Error = FemeErrorAbort;
  ierr = backends[matchidx].init(resource, *feme);FemeChk(ierr);
  return 0;
}

int FemeDestroy(Feme *feme) {
  int ierr;

  if (!*feme) return 0;
  if ((*feme)->Destroy) {
    ierr = (*feme)->Destroy(*feme);FemeChk(ierr);
  }
  ierr = FemeFree(feme);FemeChk(ierr);
  return 0;
}
