#define _POSIX_C_SOURCE 200112
#include <ceed-impl.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static CeedRequest ceed_request_immediate;
CeedRequest *CEED_REQUEST_IMMEDIATE = &ceed_request_immediate;

static struct {
  char prefix[CEED_MAX_RESOURCE_LEN];
  int (*init)(const char *resource, Ceed f);
} backends[32];
static size_t num_backends;

int CeedErrorImpl(Ceed ceed, const char *filename, int lineno, const char *func, int ecode, const char *format, ...) {
  va_list args;
  va_start(args, format);
  if (ceed) return ceed->Error(ceed, filename, lineno, func, ecode, format, args);
  return CeedErrorAbort(ceed, filename, lineno, func, ecode, format, args);
}

int CeedErrorReturn(Ceed ceed, const char *filename, int lineno, const char *func, int ecode, const char *format, va_list args) {
  return ecode;
}

int CeedErrorAbort(Ceed ceed, const char *filename, int lineno, const char *func, int ecode, const char *format, va_list args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  abort();
  return ecode;
}

int CeedRegister(const char *prefix, int (*init)(const char *resource, Ceed f)) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0])) {
    return CeedError(NULL, 1, "Too many backends");
  }
  strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
  backends[num_backends].init = init;
  num_backends++;
  return 0;
}

int CeedMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void**)p, CEED_ALIGN, n*unit);
  if (ierr) return CeedError(NULL, ierr, "posix_memalign failed to allocate %zd members of size %zd\n", n, unit);
  return 0;
}

int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void**)p = calloc(n, unit);
  if (n && unit && !*(void**)p) return CeedError(NULL, 1, "calloc failed to allocate %zd members of size %zd\n", n, unit);
  return 0;
}

// Takes void* to avoid needing a cast, but is the address of the pointer.
int CeedFree(void *p) {
  free(*(void**)p);
  *(void**)p = NULL;
  return 0;
}

int CeedInit(const char *resource, Ceed *ceed) {
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
  if (!matchlen) return CeedError(NULL, 1, "No suitable backend");
  ierr = CeedCalloc(1,ceed);CeedChk(ierr);
  (*ceed)->Error = CeedErrorAbort;
  ierr = backends[matchidx].init(resource, *ceed);CeedChk(ierr);
  return 0;
}

int CeedDestroy(Ceed *ceed) {
  int ierr;

  if (!*ceed) return 0;
  if ((*ceed)->Destroy) {
    ierr = (*ceed)->Destroy(*ceed);CeedChk(ierr);
  }
  ierr = CeedFree(ceed);CeedChk(ierr);
  return 0;
}
