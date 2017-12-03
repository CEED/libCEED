#ifndef _feme_impl_h
#define _feme_impl_h

#include <feme.h>

#define FEME_INTERN FEME_EXTERN __attribute__((visibility ("hidden")))

#define FEME_MAX_RESOURCE_LEN 1024
#define FEME_ALIGN 64

struct Feme_private {
  int (*Error)(Feme, const char *, int, const char *, int, const char *, va_list);
  int (*Destroy)(Feme);
  int (*VecCreate)(Feme, FemeInt, FemeVector);
  int (*ElemRestrictionCreate)(FemeElemRestriction, FemeMemType, FemeCopyMode, const FemeInt *);
  int (*BasisCreateTensorH1)(Feme, FemeInt, FemeInt, FemeInt, const FemeScalar *, const FemeScalar *, const FemeScalar *, const FemeScalar *, FemeBasis);
  int (*QFunctionCreate)(FemeQFunction);
  int (*OperatorCreate)(FemeOperator);
};

/* In the next 3 functions, p has to be the address of a pointer type, i.e. p
   has to be a pointer to a pointer. */
FEME_INTERN int FemeMallocArray(size_t n, size_t unit, void *p);
FEME_INTERN int FemeCallocArray(size_t n, size_t unit, void *p);
FEME_INTERN int FemeFree(void *p);

#define FemeChk(ierr) do { if (ierr) return ierr; } while (0)
#define FemeMalloc(n, p) FemeMallocArray((n), sizeof(**(p)), p)
#define FemeCalloc(n, p) FemeCallocArray((n), sizeof(**(p)), p)

struct FemeVector_private {
  Feme feme;
  int (*SetArray)(FemeVector, FemeMemType, FemeCopyMode, FemeScalar *);
  int (*GetArray)(FemeVector, FemeMemType, FemeScalar **);
  int (*GetArrayRead)(FemeVector, FemeMemType, const FemeScalar **);
  int (*RestoreArray)(FemeVector, FemeScalar **);
  int (*RestoreArrayRead)(FemeVector, const FemeScalar **);
  int (*Destroy)(FemeVector);
  FemeInt length;
  void *data;
};

struct FemeElemRestriction_private {
  Feme feme;
  int (*Apply)(FemeElemRestriction, FemeTransposeMode, FemeVector, FemeVector, FemeRequest *);
  int (*Destroy)(FemeElemRestriction);
  FemeInt nelem;    /* number of elements */
  FemeInt elemsize; /* number of dofs per element */
  FemeInt ndof;     /* size of the L-vector, can be used for checking for
                       correct vector sizes */
  void *data;       /* place for the backend to store any data */
};

/* FIXME: Since we will want to support non-tensor product bases, and other
   types, like H(div)- and H(curl)-conforming bases, separate the basis data, so
   it can be changed. In other words, replace { dim, P1d, Q1d, qref1d,
   qweight1d, interp1d, grad1d } with void *data. */
struct FemeBasis_private {
  Feme feme;
  int (*Apply)(FemeBasis, FemeTransposeMode, FemeEvalMode, const FemeScalar *, FemeScalar *);
  int (*Destroy)(FemeBasis);
  FemeInt dim;
  FemeInt ndof;
  FemeInt P1d;
  FemeInt Q1d;
  FemeScalar *qref1d;
  FemeScalar *qweight1d;
  FemeScalar *interp1d;
  FemeScalar *grad1d;
};

/* FIXME: The number of in-fields and out-fields may be different? */
/* FIXME: Shouldn't inmode and outmode be per-in-field and per-out-field,
   respectively? */
/* FIXME: Should we make this an "abstact" class, i.e. support different types
   of Q-functions, using different sets of data fields? */
struct FemeQFunction_private {
  Feme feme;
  int (*Destroy)(FemeQFunction);
  FemeInt vlength;    // Number of quadrature points must be padded to a multiple of vlength
  FemeInt nfields;
  size_t qdatasize;   // Number of bytes of qdata per quadrature point
  FemeEvalMode inmode, outmode;
  int (*function)(void*, void*, FemeInt, const FemeScalar *const*, FemeScalar *const*);
  const char *focca;
  void *ctx;      /* user context for function */
  size_t ctxsize; /* size of user context; may be used to copy to a device */
  void *data;     /* backend data */
};

/* FIXME: Should we make this an "abstact" class, i.e. support different types
   of operators, using different sets of data fields? */
struct FemeOperator_private {
  Feme feme;
  int (*Apply)(FemeOperator, FemeVector, FemeVector, FemeVector, FemeRequest*);
  int (*ApplyJacobian)(FemeOperator, FemeVector, FemeVector, FemeVector, FemeVector, FemeRequest*);
  int (*Destroy)(FemeOperator);
  FemeElemRestriction Erestrict;
  FemeBasis basis;
  FemeQFunction qf;
  FemeQFunction dqf;
  FemeQFunction dqfT;
  void *data;
};

#endif
