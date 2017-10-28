#ifndef _feme_impl_h
#define _feme_impl_h

#include <feme.h>

#define FEME_INTERN FEME_EXTERN __attribute__((visibility ("hidden")))

#define FEME_MAX_RESOURCE_LEN 1024
#define FEME_ALIGN 64

struct Feme_private {
  int (*Error)(Feme, int, const char *, va_list);
  int (*Destroy)(Feme);
};

FEME_INTERN int FemeMallocArray(size_t n, size_t unit, void *p);
FEME_INTERN int FemeCallocArray(size_t n, size_t unit, void *p);

#define FemeChk(ierr) do { if (ierr) return ierr; } while (0)
#define FemeMalloc(n, p) FemeMallocArray((n), sizeof(**(p)), p)
#define FemeCalloc(n, p) FemeCallocArray((n), sizeof(**(p)), p)
#define FemeFree(p) (free(*(p)), *(p) = NULL, 0)

#endif
