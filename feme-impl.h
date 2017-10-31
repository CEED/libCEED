#ifndef _feme_impl_h
#define _feme_impl_h

#include <feme.h>

#define FEME_INTERN FEME_EXTERN __attribute__((visibility ("hidden")))

#define FEME_MAX_RESOURCE_LEN 1024
#define FEME_ALIGN 64

struct Feme_private {
  int (*Error)(Feme, const char *, int, const char *, int, const char *, va_list);
  int (*Destroy)(Feme);
  int (*VecCreate)(Feme, FemeInt, FemeVec);
  int (*ElemRestrictionCreate)(Feme, FemeInt, FemeInt, FemeMemType, FemeCopyMode, const FemeInt *, FemeElemRestriction);
  int (*FemeBasisCreateTensorH1)(Feme, FemeInt, FemeInt, FemeInt, const FemeScalar *, const FemeScalar *, const FemeScalar *, const FemeScalar *, FemeBasis);
};

FEME_INTERN int FemeMallocArray(size_t n, size_t unit, void *p);
FEME_INTERN int FemeCallocArray(size_t n, size_t unit, void *p);
FEME_INTERN int FemeFree(void *p);

#define FemeChk(ierr) do { if (ierr) return ierr; } while (0)
#define FemeMalloc(n, p) FemeMallocArray((n), sizeof(**(p)), p)
#define FemeCalloc(n, p) FemeCallocArray((n), sizeof(**(p)), p)

struct FemeVec_private {
  Feme feme;
  int (*SetArray)(FemeVec, FemeMemType, FemeCopyMode, FemeScalar *);
  int (*GetArray)(FemeVec, FemeMemType, FemeScalar **);
  int (*GetArrayRead)(FemeVec, FemeMemType, const FemeScalar **);
  int (*RestoreArray)(FemeVec, FemeScalar **);
  int (*RestoreArrayRead)(FemeVec, const FemeScalar **);
  int (*Destroy)(FemeVec);
  FemeInt n;
  void *data;
};

struct FemeElemRestriction_private {
  Feme feme;
  FemeInt nelem;
  FemeInt elemsize;
  FemeInt ndof;
  void *data;
};

#endif
