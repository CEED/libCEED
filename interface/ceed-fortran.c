// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// Fortran interface
#include <ceed-fortran-name.h>
#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define FORTRAN_REQUEST_IMMEDIATE -1
#define FORTRAN_REQUEST_ORDERED -2
#define FORTRAN_NULL -3
#define FORTRAN_STRIDES_BACKEND -4
#define FORTRAN_VECTOR_ACTIVE -5
#define FORTRAN_VECTOR_NONE -6
#define FORTRAN_ELEMRESTRICTION_NONE -7
#define FORTRAN_BASIS_NONE -8
#define FORTRAN_QFUNCTION_NONE -9

static CeedRequest *CeedRequest_dict       = NULL;
static int          max_CeedRequest        = 0;
static int          num_CeedRequest        = 0;
static int          num_active_CeedRequest = 0;

static inline void fCeedRequestExpandDict(void) {
  max_CeedRequest += max_CeedRequest / 2 + 1;
  CeedRealloc(max_CeedRequest, &CeedRequest_dict);
}

static inline void fCeedRequestAccept(int *request) {
  *request = num_CeedRequest;
  num_CeedRequest++;
  num_active_CeedRequest++;
}

#define fCeedRequestWait FORTRAN_NAME(ceedrequestwait, CEEDREQUESTWAIT)
CEED_EXTERN void fCeedRequestWait(int *request, int *err) {
  // TODO Uncomment this once CeedRequestWait is implemented
  //*err = CeedRequestWait(&CeedRequest_dict[*request]);

  if (*err == 0) {
    num_active_CeedRequest--;
    if (num_active_CeedRequest == 0) {
      CeedFree(&CeedRequest_dict);
      num_CeedRequest = 0;
      max_CeedRequest = 0;
    }
  }
}

// This test should actually be for the gfortran version, but we don't currently
// have a configure system to determine that (TODO).  At present, this will use
// the smaller integer when run with clang+gfortran=8, for example.  (That is
// sketchy, but will likely work for users that don't have huge character
// strings.)
#if __GNUC__ >= 8
typedef size_t fortran_charlen_t;
#else
typedef int fortran_charlen_t;
#endif

// Fortran strings are generally unterminated and the length is passed as an
// extra argument after all the normal arguments.  Some compilers (I only know
// of Windows) place the length argument immediately after the string parameter
// (TODO).
//
// We can't just NULL-terminate the string in-place because that could overwrite
// other strings or attempt to write to read-only memory.  This macro allocates
// a string to hold the null-terminated version of the string that C expects.
#define CEED_MAX_FORTRAN_STRING_LEN 1024
#define FIX_STRING(stringname)                                                             \
  char stringname##_c[CEED_MAX_FORTRAN_STRING_LEN] = {'\0'};                               \
  if (stringname##_len > CEED_MAX_FORTRAN_STRING_LEN - 1) {                                \
    *err = CeedError(NULL, 1, "Fortran string too long: %zd", (size_t)(stringname##_len)); \
  }                                                                                        \
  strncpy(stringname##_c, stringname, CeedIntMin(stringname##_len, CEED_MAX_FORTRAN_STRING_LEN - 1));

// -----------------------------------------------------------------------------
// Ceed
// -----------------------------------------------------------------------------
static Ceed *Ceed_dict       = NULL;
static int   max_Ceed        = 0;
static int   num_Ceed        = 0;
static int   num_active_Ceed = 0;

static inline void fCeedExpandDict(void) {
  max_Ceed += max_Ceed / 2 + 1;
  CeedRealloc(max_Ceed, &Ceed_dict);
}

static inline void fCeedAccept(int *ceed) {
  *ceed = num_Ceed;
  num_Ceed++;
  num_active_Ceed++;
}

#define fCeedInit FORTRAN_NAME(ceedinit, CEEDINIT)
CEED_EXTERN void fCeedInit(const char *resource, int *ceed, int *err, fortran_charlen_t resource_len) {
  FIX_STRING(resource);
  if (num_Ceed == max_Ceed) fCeedExpandDict();
  *err = CeedInit(resource_c, &Ceed_dict[num_Ceed]);
  if (*err == 0) fCeedAccept(ceed);
}

#define fCeedIsDeterministic FORTRAN_NAME(ceedisdeterministic, CEEDISDETERMINISTIC)
CEED_EXTERN void fCeedIsDeterministic(int *ceed, int *is_deterministic, int *err) {
  *err = CeedIsDeterministic(Ceed_dict[*ceed], (bool *)is_deterministic);
}

#define fCeedGetPreferredMemType FORTRAN_NAME(ceedgetpreferredmemtype, CEEDGETPREFERREDMEMTYPE)
CEED_EXTERN void fCeedGetPreferredMemType(int *ceed, int *type, int *err) { *err = CeedGetPreferredMemType(Ceed_dict[*ceed], (CeedMemType *)type); }

#define fCeedSetNumViewTabs FORTRAN_NAME(ceedsetnumviewtabs, CEEDSETNUMVIEWTABS)
CEED_EXTERN void fCeedSetNumViewTabs(int *ceed, int *num_tabs, int *err) { *err = CeedSetNumViewTabs(Ceed_dict[*ceed], *num_tabs); }

#define fCeedView FORTRAN_NAME(ceedview, CEEDVIEW)
CEED_EXTERN void fCeedView(int *ceed, int *err) { *err = CeedView(Ceed_dict[*ceed], stdout); }

#define fCeedDestroy FORTRAN_NAME(ceeddestroy, CEEDDESTROY)
CEED_EXTERN void fCeedDestroy(int *ceed, int *err) {
  if (*ceed == FORTRAN_NULL) return;
  *err = CeedDestroy(&Ceed_dict[*ceed]);

  if (*err == 0) {
    *ceed = FORTRAN_NULL;
    num_active_Ceed--;
    if (num_active_Ceed == 0) {
      CeedFree(&Ceed_dict);
      num_Ceed = 0;
      max_Ceed = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedVector
// -----------------------------------------------------------------------------
static CeedVector *CeedVector_dict       = NULL;
static int         max_CeedVector        = 0;
static int         num_CeedVector        = 0;
static int         num_active_CeedVector = 0;

static inline void fCeedVectorExpandDict(void) {
  max_CeedVector += max_CeedVector / 2 + 1;
  CeedRealloc(max_CeedVector, &CeedVector_dict);
}

static inline void fCeedVectorAccept(int *vec) {
  *vec = num_CeedVector;
  num_CeedVector++;
  num_active_CeedVector++;
}

#define fCeedVectorCreate FORTRAN_NAME(ceedvectorcreate, CEEDVECTORCREATE)
CEED_EXTERN void fCeedVectorCreate(int *ceed, int *length, int *vec, int *err) {
  if (num_CeedVector == max_CeedVector) fCeedVectorExpandDict();
  *err = CeedVectorCreate(Ceed_dict[*ceed], *length, &CeedVector_dict[num_CeedVector]);
  if (*err == 0) fCeedVectorAccept(vec);
}

#define fCeedVectorSetArray FORTRAN_NAME(ceedvectorsetarray, CEEDVECTORSETARRAY)
CEED_EXTERN void fCeedVectorSetArray(int *vec, int *mem_type, int *copy_mode, CeedScalar *array, int64_t *offset, int *err) {
  *err = CeedVectorSetArray(CeedVector_dict[*vec], (CeedMemType)*mem_type, (CeedCopyMode)*copy_mode, (CeedScalar *)(array + *offset));
}

#define fCeedVectorTakeArray FORTRAN_NAME(ceedvectortakearray, CEEDVECTORTAKEARRAY)
CEED_EXTERN void fCeedVectorTakeArray(int *vec, int *mem_type, CeedScalar *array, int64_t *offset, int *err) {
  CeedScalar *array_c;

  *err    = CeedVectorTakeArray(CeedVector_dict[*vec], (CeedMemType)*mem_type, &array_c);
  *offset = array_c - array;
}

#define fCeedVectorSyncArray FORTRAN_NAME(ceedvectorsyncarray, CEEDVECTORSYNCARRAY)
CEED_EXTERN void fCeedVectorSyncArray(int *vec, int *mem_type, int *err) {
  *err = CeedVectorSyncArray(CeedVector_dict[*vec], (CeedMemType)*mem_type);
}

#define fCeedVectorSetValue FORTRAN_NAME(ceedvectorsetvalue, CEEDVECTORSETVALUE)
CEED_EXTERN void fCeedVectorSetValue(int *vec, CeedScalar *value, int *err) { *err = CeedVectorSetValue(CeedVector_dict[*vec], *value); }

#define fCeedVectorGetArray FORTRAN_NAME(ceedvectorgetarray, CEEDVECTORGETARRAY)
CEED_EXTERN void fCeedVectorGetArray(int *vec, int *mem_type, CeedScalar *array, int64_t *offset, int *err) {
  CeedScalar *array_c;

  *err    = CeedVectorGetArray(CeedVector_dict[*vec], (CeedMemType)*mem_type, &array_c);
  *offset = array_c - array;
}

#define fCeedVectorGetArrayRead FORTRAN_NAME(ceedvectorgetarrayread, CEEDVECTORGETARRAYREAD)
CEED_EXTERN void fCeedVectorGetArrayRead(int *vec, int *mem_type, CeedScalar *array, int64_t *offset, int *err) {
  const CeedScalar *array_c;

  *err    = CeedVectorGetArrayRead(CeedVector_dict[*vec], (CeedMemType)*mem_type, &array_c);
  *offset = array_c - array;
}

#define fCeedVectorGetArrayWrite FORTRAN_NAME(ceedvectorgetarraywrite, CEEDVECTORGETARRAYWRITE)
CEED_EXTERN void fCeedVectorGetArrayWrite(int *vec, int *mem_type, CeedScalar *array, int64_t *offset, int *err) {
  CeedScalar *array_c;

  *err    = CeedVectorGetArrayWrite(CeedVector_dict[*vec], (CeedMemType)*mem_type, &array_c);
  *offset = array_c - array;
}

#define fCeedVectorRestoreArray FORTRAN_NAME(ceedvectorrestorearray, CEEDVECTORRESTOREARRAY)
CEED_EXTERN void fCeedVectorRestoreArray(int *vec, CeedScalar *array, int64_t *offset, int *err) {
  CeedScalar *array_offset = array + *offset;

  *err    = CeedVectorRestoreArray(CeedVector_dict[*vec], &array_offset);
  *offset = 0;
}

#define fCeedVectorRestoreArrayRead FORTRAN_NAME(ceedvectorrestorearrayread, CEEDVECTORRESTOREARRAYREAD)
CEED_EXTERN void fCeedVectorRestoreArrayRead(int *vec, const CeedScalar *array, int64_t *offset, int *err) {
  *err    = CeedVectorRestoreArrayRead(CeedVector_dict[*vec], &array);
  *offset = 0;
}

#define fCeedVectorNorm FORTRAN_NAME(ceedvectornorm, CEEDVECTORNORM)
CEED_EXTERN void fCeedVectorNorm(int *vec, int *norm_type, CeedScalar *norm, int *err) {
  *err = CeedVectorNorm(CeedVector_dict[*vec], (CeedNormType)*norm_type, norm);
}

#define fCeedVectorReciprocal FORTRAN_NAME(ceedvectorreciprocal, CEEDVECTORRECIPROCAL)
CEED_EXTERN void fCeedVectorReciprocal(int *vec, int *err) { *err = CeedVectorReciprocal(CeedVector_dict[*vec]); }

#define fCeedVectorSetNumViewTabs FORTRAN_NAME(ceedvectorsetnumviewtabs, CEEDVECTORSETNUMVIEWTABS)
CEED_EXTERN void fCeedVectorSetNumViewTabs(int *vec, int *num_tabs, int *err) { *err = CeedVectorSetNumViewTabs(CeedVector_dict[*vec], *num_tabs); }

#define fCeedVectorView FORTRAN_NAME(ceedvectorview, CEEDVECTORVIEW)
CEED_EXTERN void fCeedVectorView(int *vec, int *err) { *err = CeedVectorView(CeedVector_dict[*vec], "%12.8f", stdout); }

#define fCeedVectorDestroy FORTRAN_NAME(ceedvectordestroy, CEEDVECTORDESTROY)
CEED_EXTERN void fCeedVectorDestroy(int *vec, int *err) {
  if (*vec == FORTRAN_NULL) return;
  *err = CeedVectorDestroy(&CeedVector_dict[*vec]);

  if (*err == 0) {
    *vec = FORTRAN_NULL;
    num_active_CeedVector--;
    if (num_active_CeedVector == 0) {
      CeedFree(&CeedVector_dict);
      num_CeedVector = 0;
      max_CeedVector = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedElemRestriction
// -----------------------------------------------------------------------------
static CeedElemRestriction *CeedElemRestriction_dict       = NULL;
static int                  max_CeedElemRestriction        = 0;
static int                  num_CeedElemRestriction        = 0;
static int                  num_active_CeedElemRestriction = 0;

static inline void fCeedElemRestrictionExpandDict(void) {
  max_CeedElemRestriction += max_CeedElemRestriction / 2 + 1;
  CeedRealloc(max_CeedElemRestriction, &CeedElemRestriction_dict);
}

static inline void fCeedElemRestrictionAccept(int *rstr) {
  *rstr = num_CeedElemRestriction;
  num_CeedElemRestriction++;
  num_active_CeedElemRestriction++;
}

#define fCeedElemRestrictionCreate FORTRAN_NAME(ceedelemrestrictioncreate, CEEDELEMRESTRICTIONCREATE)
CEED_EXTERN void fCeedElemRestrictionCreate(int *ceed, int *num_elem, int *elem_size, int *num_comp, int *comp_stride, int *l_vec_size, int *mem_type,
                                            int *copy_mode, const int *offsets, int *rstr, int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();

  const int *offsets_c = offsets;

  *err = CeedElemRestrictionCreate(Ceed_dict[*ceed], *num_elem, *elem_size, *num_comp, *comp_stride, *l_vec_size, (CeedMemType)*mem_type,
                                   (CeedCopyMode)*copy_mode, offsets_c, &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionCreateOriented FORTRAN_NAME(ceedelemrestrictioncreateoriented, CEEDELEMRESTRICTIONCREATEORIENTED)
CEED_EXTERN void fCeedElemRestrictionCreateOriented(int *ceed, int *num_elem, int *elem_size, int *num_comp, int *comp_stride, int *l_vec_size,
                                                    int *mem_type, int *copy_mode, const int *offsets, const bool *orients, int *rstr, int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();

  const int  *offsets_c = offsets;
  const bool *orients_c = orients;

  *err = CeedElemRestrictionCreateOriented(Ceed_dict[*ceed], *num_elem, *elem_size, *num_comp, *comp_stride, *l_vec_size, (CeedMemType)*mem_type,
                                           (CeedCopyMode)*copy_mode, offsets_c, orients_c, &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionCreateCurlOriented FORTRAN_NAME(ceedelemrestrictioncreatecurloriented, CEEDELEMRESTRICTIONCREATECURLORIENTED)
CEED_EXTERN void fCeedElemRestrictionCreateCurlOriented(int *ceed, int *num_elem, int *elem_size, int *num_comp, int *comp_stride, int *l_vec_size,
                                                        int *mem_type, int *copy_mode, const int *offsets, const int8_t *curl_orients, int *rstr,
                                                        int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();

  const int    *offsets_c      = offsets;
  const int8_t *curl_orients_c = curl_orients;

  *err =
      CeedElemRestrictionCreateCurlOriented(Ceed_dict[*ceed], *num_elem, *elem_size, *num_comp, *comp_stride, *l_vec_size, (CeedMemType)*mem_type,
                                            (CeedCopyMode)*copy_mode, offsets_c, curl_orients_c, &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionCreateStrided FORTRAN_NAME(ceedelemrestrictioncreatestrided, CEEDELEMRESTRICTIONCREATESTRIDED)
CEED_EXTERN void fCeedElemRestrictionCreateStrided(int *ceed, int *num_elem, int *elem_size, int *num_comp, int *l_vec_size, int *strides, int *rstr,
                                                   int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();
  *err = CeedElemRestrictionCreateStrided(Ceed_dict[*ceed], *num_elem, *elem_size, *num_comp, *l_vec_size,
                                          *strides == FORTRAN_STRIDES_BACKEND ? CEED_STRIDES_BACKEND : strides,
                                          &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionCreateBlocked FORTRAN_NAME(ceedelemrestrictioncreateblocked, CEEDELEMRESTRICTIONCREATEBLOCKED)
CEED_EXTERN void fCeedElemRestrictionCreateBlocked(int *ceed, int *num_elem, int *elem_size, int *block_size, int *num_comp, int *comp_stride,
                                                   int *l_vec_size, int *mem_type, int *copy_mode, const int *offsets, int *rstr, int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();

  const int *offsets_c = offsets;

  *err = CeedElemRestrictionCreateBlocked(Ceed_dict[*ceed], *num_elem, *elem_size, *block_size, *num_comp, *comp_stride, *l_vec_size,
                                          (CeedMemType)*mem_type, (CeedCopyMode)*copy_mode, offsets_c,
                                          &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionCreateBlockedOriented FORTRAN_NAME(ceedelemrestrictioncreateblockedoriented, CEEDELEMRESTRICTIONCREATEBLOCKEDORIENTED)
CEED_EXTERN void fCeedElemRestrictionCreateBlockedOriented(int *ceed, int *num_elem, int *elem_size, int *block_size, int *num_comp, int *comp_stride,
                                                           int *l_vec_size, int *mem_type, int *copy_mode, const int *offsets, const bool *orients,
                                                           int *rstr, int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();

  const int  *offsets_c = offsets;
  const bool *orients_c = orients;

  *err = CeedElemRestrictionCreateBlockedOriented(Ceed_dict[*ceed], *num_elem, *elem_size, *block_size, *num_comp, *comp_stride, *l_vec_size,
                                                  (CeedMemType)*mem_type, (CeedCopyMode)*copy_mode, offsets_c, orients_c,
                                                  &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionCreateBlockedCurlOriented \
  FORTRAN_NAME(ceedelemrestrictioncreateblockedcurloriented, CEEDELEMRESTRICTIONCREATEBLOCKEDCURLORIENTED)
CEED_EXTERN void fCeedElemRestrictionCreateBlockedCurlOriented(int *ceed, int *num_elem, int *elem_size, int *block_size, int *num_comp,
                                                               int *comp_stride, int *l_vec_size, int *mem_type, int *copy_mode, const int *offsets,
                                                               const int8_t *curl_orients, int *rstr, int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();

  const int    *offsets_c      = offsets;
  const int8_t *curl_orients_c = curl_orients;

  *err = CeedElemRestrictionCreateBlockedCurlOriented(Ceed_dict[*ceed], *num_elem, *elem_size, *block_size, *num_comp, *comp_stride, *l_vec_size,
                                                      (CeedMemType)*mem_type, (CeedCopyMode)*copy_mode, offsets_c, curl_orients_c,
                                                      &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionCreateBlockedStrided FORTRAN_NAME(ceedelemrestrictioncreateblockedstrided, CEEDELEMRESTRICTIONCREATEBLOCKEDSTRIDED)
CEED_EXTERN void fCeedElemRestrictionCreateBlockedStrided(int *ceed, int *num_elem, int *elem_size, int *block_size, int *num_comp, int *l_vec_size,
                                                          int *strides, int *rstr, int *err) {
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();
  *err = CeedElemRestrictionCreateBlockedStrided(Ceed_dict[*ceed], *num_elem, *elem_size, *block_size, *num_comp, *l_vec_size, strides,
                                                 &CeedElemRestriction_dict[num_CeedElemRestriction]);
  if (*err == 0) fCeedElemRestrictionAccept(rstr);
}

#define fCeedElemRestrictionApply FORTRAN_NAME(ceedelemrestrictionapply, CEEDELEMRESTRICTIONAPPLY)
CEED_EXTERN void fCeedElemRestrictionApply(int *rstr, int *t_mode, int *u, int *ru, int *request, int *err) {
  bool create_request = true;

  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*request == FORTRAN_REQUEST_IMMEDIATE || *request == FORTRAN_REQUEST_ORDERED) create_request = false;
  if (create_request && num_CeedRequest == max_CeedRequest) fCeedRequestExpandDict();

  CeedRequest *request_c;

  if (*request == FORTRAN_REQUEST_IMMEDIATE) {
    request_c = CEED_REQUEST_IMMEDIATE;
  } else if (*request == FORTRAN_REQUEST_ORDERED) {
    request_c = CEED_REQUEST_ORDERED;
  } else {
    request_c = &CeedRequest_dict[num_CeedRequest];
  }

  *err = CeedElemRestrictionApply(CeedElemRestriction_dict[*rstr], (CeedTransposeMode)*t_mode, CeedVector_dict[*u], CeedVector_dict[*ru], request_c);
  if (*err == 0 && create_request) fCeedRequestAccept(request);
}

#define fCeedElemRestrictionApplyBlock FORTRAN_NAME(ceedelemrestrictionapplyblock, CEEDELEMRESTRICTIONAPPLYBLOCK)
CEED_EXTERN void fCeedElemRestrictionApplyBlock(int *rstr, int *block, int *t_mode, int *u, int *ru, int *request, int *err) {
  bool create_request = true;

  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*request == FORTRAN_REQUEST_IMMEDIATE || *request == FORTRAN_REQUEST_ORDERED) create_request = false;
  if (create_request && num_CeedRequest == max_CeedRequest) fCeedRequestExpandDict();

  CeedRequest *request_c;

  if (*request == FORTRAN_REQUEST_IMMEDIATE) {
    request_c = CEED_REQUEST_IMMEDIATE;
  } else if (*request == FORTRAN_REQUEST_ORDERED) {
    request_c = CEED_REQUEST_ORDERED;
  } else {
    request_c = &CeedRequest_dict[num_CeedRequest];
  }

  *err = CeedElemRestrictionApplyBlock(CeedElemRestriction_dict[*rstr], *block, (CeedTransposeMode)*t_mode, CeedVector_dict[*u], CeedVector_dict[*ru],
                                       request_c);
  if (*err == 0 && create_request) fCeedRequestAccept(request);
}

#define fCeedElemRestrictionGetMultiplicity FORTRAN_NAME(ceedelemrestrictiongetmultiplicity, CEEDELEMRESTRICTIONGETMULTIPLICITY)
CEED_EXTERN void fCeedElemRestrictionGetMultiplicity(int *rstr, int *mult, int *err) {
  *err = CeedElemRestrictionGetMultiplicity(CeedElemRestriction_dict[*rstr], CeedVector_dict[*mult]);
}

#define fCeedElemRestrictionGetELayout FORTRAN_NAME(ceedelemrestrictiongetelayout, CEEDELEMRESTRICTIONGETELAYOUT)
CEED_EXTERN void fCeedElemRestrictionGetELayout(int *rstr, int *layout, int *err) {
  CeedInt layout_c[3];

  *err = CeedElemRestrictionGetELayout(CeedElemRestriction_dict[*rstr], layout_c);
  for (int i = 0; i < 3; i++) layout[i] = layout_c[i];
}

#define fCeedElemRestrictionSetNumViewTabs FORTRAN_NAME(ceedelemrestrictionsetnumviewtabs, CEEDELEMRESTRICTIONSETNUMVIEWTABS)
CEED_EXTERN void fCeedElemRestrictionSetNumViewTabs(int *rstr, int *num_tabs, int *err) {
  *err = CeedElemRestrictionSetNumViewTabs(CeedElemRestriction_dict[*rstr], *num_tabs);
}

#define fCeedElemRestrictionView FORTRAN_NAME(ceedelemrestrictionview, CEEDELEMRESTRICTIONVIEW)
CEED_EXTERN void fCeedElemRestrictionView(int *rstr, int *err) { *err = CeedElemRestrictionView(CeedElemRestriction_dict[*rstr], stdout); }

#define fCeedElemRestrictionDestroy FORTRAN_NAME(ceedelemrestrictiondestroy, CEEDELEMRESTRICTIONDESTROY)
CEED_EXTERN void fCeedElemRestrictionDestroy(int *rstr, int *err) {
  if (*rstr == FORTRAN_NULL) return;
  *err = CeedElemRestrictionDestroy(&CeedElemRestriction_dict[*rstr]);

  if (*err == 0) {
    *rstr = FORTRAN_NULL;
    num_active_CeedElemRestriction--;
    if (num_active_CeedElemRestriction == 0) {
      CeedFree(&CeedElemRestriction_dict);
      num_CeedElemRestriction = 0;
      max_CeedElemRestriction = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedBasis
// -----------------------------------------------------------------------------
static CeedBasis *CeedBasis_dict       = NULL;
static int        max_CeedBasis        = 0;
static int        num_CeedBasis        = 0;
static int        num_active_CeedBasis = 0;

static inline void fCeedBasisExpandDict(void) {
  max_CeedBasis += max_CeedBasis / 2 + 1;
  CeedRealloc(max_CeedBasis, &CeedBasis_dict);
}

static inline void fCeedBasisAccept(int *basis) {
  *basis = num_CeedBasis;
  num_CeedBasis++;
  num_active_CeedBasis++;
}

#define fCeedBasisCreateTensorH1Lagrange FORTRAN_NAME(ceedbasiscreatetensorh1lagrange, CEEDBASISCREATETENSORH1LAGRANGE)
CEED_EXTERN void fCeedBasisCreateTensorH1Lagrange(int *ceed, int *dim, int *num_comp, int *P, int *Q, int *quad_mode, int *basis, int *err) {
  if (num_CeedBasis == max_CeedBasis) fCeedBasisExpandDict();
  *err = CeedBasisCreateTensorH1Lagrange(Ceed_dict[*ceed], *dim, *num_comp, *P, *Q, (CeedQuadMode)*quad_mode, &CeedBasis_dict[num_CeedBasis]);
  if (*err == 0) fCeedBasisAccept(basis);
}

#define fCeedBasisCreateTensorH1 FORTRAN_NAME(ceedbasiscreatetensorh1, CEEDBASISCREATETENSORH1)
CEED_EXTERN void fCeedBasisCreateTensorH1(int *ceed, int *dim, int *num_comp, int *P_1d, int *Q_1d, const CeedScalar *interp_1d,
                                          const CeedScalar *grad_1d, const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, int *basis,
                                          int *err) {
  if (num_CeedBasis == max_CeedBasis) fCeedBasisExpandDict();
  *err = CeedBasisCreateTensorH1(Ceed_dict[*ceed], *dim, *num_comp, *P_1d, *Q_1d, interp_1d, grad_1d, q_ref_1d, q_weight_1d,
                                 &CeedBasis_dict[num_CeedBasis]);
  if (*err == 0) fCeedBasisAccept(basis);
}

#define fCeedBasisCreateH1 FORTRAN_NAME(ceedbasiscreateh1, CEEDBASISCREATEH1)
CEED_EXTERN void fCeedBasisCreateH1(int *ceed, int *topo, int *num_comp, int *num_nodes, int *num_qpts, const CeedScalar *interp,
                                    const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, int *basis, int *err) {
  if (num_CeedBasis == max_CeedBasis) fCeedBasisExpandDict();
  *err = CeedBasisCreateH1(Ceed_dict[*ceed], (CeedElemTopology)*topo, *num_comp, *num_nodes, *num_qpts, interp, grad, q_ref, q_weight,
                           &CeedBasis_dict[num_CeedBasis]);
  if (*err == 0) fCeedBasisAccept(basis);
}

#define fCeedBasisCreateHdiv FORTRAN_NAME(ceedbasiscreatehdiv, CEEDBASISCREATEHDIV)
CEED_EXTERN void fCeedBasisCreateHdiv(int *ceed, int *topo, int *num_comp, int *num_nodes, int *num_qpts, const CeedScalar *interp,
                                      const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, int *basis, int *err) {
  if (num_CeedBasis == max_CeedBasis) fCeedBasisExpandDict();
  *err = CeedBasisCreateHdiv(Ceed_dict[*ceed], (CeedElemTopology)*topo, *num_comp, *num_nodes, *num_qpts, interp, div, q_ref, q_weight,
                             &CeedBasis_dict[num_CeedBasis]);
  if (*err == 0) fCeedBasisAccept(basis);
}

#define fCeedBasisCreateHcurl FORTRAN_NAME(ceedbasiscreatehcurl, CEEDBASISCREATEHCURL)
CEED_EXTERN void fCeedBasisCreateHcurl(int *ceed, int *topo, int *num_comp, int *num_nodes, int *num_qpts, const CeedScalar *interp,
                                       const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, int *basis, int *err) {
  if (num_CeedBasis == max_CeedBasis) fCeedBasisExpandDict();
  *err = CeedBasisCreateHcurl(Ceed_dict[*ceed], (CeedElemTopology)*topo, *num_comp, *num_nodes, *num_qpts, interp, curl, q_ref, q_weight,
                              &CeedBasis_dict[num_CeedBasis]);
  if (*err == 0) fCeedBasisAccept(basis);
}

#define fCeedBasisSetNumViewTabs FORTRAN_NAME(ceedbasissetnumviewtabs, CEEDBASISSETNUMVIEWTABS)
CEED_EXTERN void fCeedBasisSetNumViewTabs(int *basis, int *num_tabs, int *err) { *err = CeedBasisSetNumViewTabs(CeedBasis_dict[*basis], *num_tabs); }

#define fCeedBasisView FORTRAN_NAME(ceedbasisview, CEEDBASISVIEW)
CEED_EXTERN void fCeedBasisView(int *basis, int *err) { *err = CeedBasisView(CeedBasis_dict[*basis], stdout); }

#define fCeedBasisGetCollocatedGrad FORTRAN_NAME(ceedbasisgetcollocatedgrad, CEEDBASISGETCOLLOCATEDGRAD)
CEED_EXTERN void fCeedBasisGetCollocatedGrad(int *basis, CeedScalar *colo_grad_1d, int *err) {
  *err = CeedBasisGetCollocatedGrad(CeedBasis_dict[*basis], colo_grad_1d);
}

#define fCeedBasisApply FORTRAN_NAME(ceedbasisapply, CEEDBASISAPPLY)
CEED_EXTERN void fCeedBasisApply(int *basis, int *num_elem, int *t_mode, int *eval_mode, int *u, int *v, int *err) {
  *err = CeedBasisApply(CeedBasis_dict[*basis], *num_elem, (CeedTransposeMode)*t_mode, (CeedEvalMode)*eval_mode,
                        *u == FORTRAN_VECTOR_NONE ? CEED_VECTOR_NONE : CeedVector_dict[*u], CeedVector_dict[*v]);
}

#define fCeedBasisGetNumNodes FORTRAN_NAME(ceedbasisgetnumnodes, CEEDBASISGETNUMNODES)
CEED_EXTERN void fCeedBasisGetNumNodes(int *basis, int *num_nodes, int *err) { *err = CeedBasisGetNumNodes(CeedBasis_dict[*basis], num_nodes); }

#define fCeedBasisGetNumQuadraturePoints FORTRAN_NAME(ceedbasisgetnumquadraturepoints, CEEDBASISGETNUMQUADRATUREPOINTS)
CEED_EXTERN void fCeedBasisGetNumQuadraturePoints(int *basis, int *num_q_pts, int *err) {
  *err = CeedBasisGetNumQuadraturePoints(CeedBasis_dict[*basis], num_q_pts);
}

#define fCeedBasisGetInterp1D FORTRAN_NAME(ceedbasisgetinterp1d, CEEDBASISGETINTERP1D)
CEED_EXTERN void fCeedBasisGetInterp1D(int *basis, CeedScalar *interp_1d, int64_t *offset, int *err) {
  const CeedScalar *interp_1d_c;

  *err    = CeedBasisGetInterp1D(CeedBasis_dict[*basis], &interp_1d_c);
  *offset = interp_1d_c - interp_1d;
}

#define fCeedBasisGetGrad1D FORTRAN_NAME(ceedbasisgetgrad1d, CEEDBASISGETGRAD1D)
CEED_EXTERN void fCeedBasisGetGrad1D(int *basis, CeedScalar *grad_1d, int64_t *offset, int *err) {
  const CeedScalar *grad_1d_c;

  *err    = CeedBasisGetGrad1D(CeedBasis_dict[*basis], &grad_1d_c);
  *offset = grad_1d_c - grad_1d;
}

#define fCeedBasisGetQRef FORTRAN_NAME(ceedbasisgetqref, CEEDBASISGETQREF)
CEED_EXTERN void fCeedBasisGetQRef(int *basis, CeedScalar *q_ref, int64_t *offset, int *err) {
  const CeedScalar *q_ref_c;

  *err    = CeedBasisGetQRef(CeedBasis_dict[*basis], &q_ref_c);
  *offset = q_ref_c - q_ref;
}

#define fCeedBasisDestroy FORTRAN_NAME(ceedbasisdestroy, CEEDBASISDESTROY)
CEED_EXTERN void fCeedBasisDestroy(int *basis, int *err) {
  if (*basis == FORTRAN_NULL) return;
  *err = CeedBasisDestroy(&CeedBasis_dict[*basis]);

  if (*err == 0) {
    *basis = FORTRAN_NULL;
    num_active_CeedBasis--;
    if (num_active_CeedBasis == 0) {
      CeedFree(&CeedBasis_dict);
      num_CeedBasis = 0;
      max_CeedBasis = 0;
    }
  }
}

#define fCeedGaussQuadrature FORTRAN_NAME(ceedgaussquadrature, CEEDGAUSSQUADRATURE)
CEED_EXTERN void fCeedGaussQuadrature(int *Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d, int *err) {
  *err = CeedGaussQuadrature(*Q, q_ref_1d, q_weight_1d);
}

#define fCeedLobattoQuadrature FORTRAN_NAME(ceedlobattoquadrature, CEEDLOBATTOQUADRATURE)
CEED_EXTERN void fCeedLobattoQuadrature(int *Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d, int *err) {
  *err = CeedLobattoQuadrature(*Q, q_ref_1d, q_weight_1d);
}

// -----------------------------------------------------------------------------
// CeedQFunctionContext
// -----------------------------------------------------------------------------
static CeedQFunctionContext *CeedQFunctionContext_dict       = NULL;
static int                   max_CeedQFunctionContext        = 0;
static int                   num_CeedQFunctionContext        = 0;
static int                   num_active_CeedQFunctionContext = 0;

static inline void fCeedQFunctionContextExpandDict(void) {
  max_CeedQFunctionContext += max_CeedQFunctionContext / 2 + 1;
  CeedRealloc(max_CeedQFunctionContext, &CeedQFunctionContext_dict);
}

static inline void fCeedQFunctionContextAccept(int *ctx) {
  *ctx = num_CeedQFunctionContext;
  num_CeedQFunctionContext++;
  num_active_CeedQFunctionContext++;
}

#define fCeedQFunctionContextCreate FORTRAN_NAME(ceedqfunctioncontextcreate, CEEDQFUNCTIONCONTEXTCREATE)
CEED_EXTERN void fCeedQFunctionContextCreate(int *ceed, int *ctx, int *err) {
  if (num_CeedQFunctionContext == max_CeedQFunctionContext) fCeedQFunctionContextExpandDict();
  *err = CeedQFunctionContextCreate(Ceed_dict[*ceed], &CeedQFunctionContext_dict[num_CeedQFunctionContext]);
  if (*err == 0) fCeedQFunctionContextAccept(ctx);
}

#define fCeedQFunctionContextSetData FORTRAN_NAME(ceedqfunctioncontextsetdata, CEEDQFUNCTIONCONTEXTSETDATA)
CEED_EXTERN void fCeedQFunctionContextSetData(int *ctx, int *mem_type, int *copy_mode, CeedInt *n, CeedScalar *data, int64_t *offset, int *err) {
  size_t ctx_size = ((size_t)*n) * sizeof(CeedScalar);

  *err = CeedQFunctionContextSetData(CeedQFunctionContext_dict[*ctx], (CeedMemType)*mem_type, (CeedCopyMode)*copy_mode, ctx_size, data + *offset);
}

#define fCeedQFunctionContextGetData FORTRAN_NAME(ceedqfunctioncontextgetdata, CEEDQFUNCTIONCONTEXTGETDATA)
CEED_EXTERN void fCeedQFunctionContextGetData(int *ctx, int *mem_type, CeedScalar *data, int64_t *offset, int *err) {
  CeedScalar *data_c;

  *err    = CeedQFunctionContextGetData(CeedQFunctionContext_dict[*ctx], (CeedMemType)*mem_type, &data_c);
  *offset = data_c - data;
}

#define fCeedQFunctionContextRestoreData FORTRAN_NAME(ceedqfunctioncontextrestoredata, CEEDQFUNCTIONCONTEXTRESTOREDATA)
CEED_EXTERN void fCeedQFunctionContextRestoreData(int *ctx, CeedScalar *data, int64_t *offset, int *err) {
  *err    = CeedQFunctionContextRestoreData(CeedQFunctionContext_dict[*ctx], (void **)&data);
  *offset = 0;
}

#define fCeedQFunctionContextSetNumViewTabs FORTRAN_NAME(ceedqfunctioncontextsetnumviewtabs, CEEDQFUNCTIONCONTEXTSETNUMVIEWTABS)
CEED_EXTERN void fCeedQFunctionContextSetNumViewTabs(int *ctx, int *num_tabs, int *err) {
  *err = CeedQFunctionContextSetNumViewTabs(CeedQFunctionContext_dict[*ctx], *num_tabs);
}

#define fCeedQFunctionContextView FORTRAN_NAME(ceedqfunctioncontextview, CEEDQFUNCTIONCONTEXTVIEW)
CEED_EXTERN void fCeedQFunctionContextView(int *ctx, int *err) { *err = CeedQFunctionContextView(CeedQFunctionContext_dict[*ctx], stdout); }

#define fCeedQFunctionContextDestroy FORTRAN_NAME(ceedqfunctioncontextdestroy, CEEDQFUNCTIONCONTEXTDESTROY)
CEED_EXTERN void fCeedQFunctionContextDestroy(int *ctx, int *err) {
  if (*ctx == FORTRAN_NULL) return;
  *err = CeedQFunctionContextDestroy(&CeedQFunctionContext_dict[*ctx]);

  if (*err == 0) {
    *ctx = FORTRAN_NULL;
    num_active_CeedQFunctionContext--;
    if (num_active_CeedQFunctionContext == 0) {
      CeedFree(&CeedQFunctionContext_dict);
      num_CeedQFunctionContext = 0;
      max_CeedQFunctionContext = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedQFunction
// -----------------------------------------------------------------------------
static CeedQFunction *CeedQFunction_dict       = NULL;
static int            num_CeedQFunction        = 0;
static int            num_active_CeedQFunction = 0;
static int            max_CeedQFunction        = 0;

static inline void fCeedQFunctionExpandDict(void) {
  max_CeedQFunction += max_CeedQFunction / 2 + 1;
  CeedRealloc(max_CeedQFunction, &CeedQFunction_dict);
}

static inline void fCeedQFunctionAccept(int *qf) {
  *qf = num_CeedQFunction;
  num_CeedQFunction++;
  num_active_CeedQFunction++;
}

// Note: Device backends are generating their own kernels from single source files, so only Host backends need to use this Fortran stub
static int CeedQFunctionFortranStub(void *ctx, int num_qpts, const CeedScalar *const *u, CeedScalar *const *v) {
  int                  ierr;
  CeedFortranContext   ctx_data  = ctx;
  CeedQFunctionContext inner_ctx = ctx_data->inner_ctx;
  CeedScalar          *ctx_f     = NULL;

  if (inner_ctx) {
    ierr = CeedQFunctionContextGetData(inner_ctx, CEED_MEM_HOST, &ctx_f);
    CeedCall(ierr);
  }
  ctx_data->f((void *)ctx_f, &num_qpts, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15], v[0],
              v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], &ierr);
  if (inner_ctx) {
    ierr = CeedQFunctionContextRestoreData(inner_ctx, &ctx_f);
    CeedCall(ierr);
  }
  return ierr;
}

#define fCeedQFunctionCreateInterior FORTRAN_NAME(ceedqfunctioncreateinterior, CEEDQFUNCTIONCREATEINTERIOR)
CEED_EXTERN void fCeedQFunctionCreateInterior(
    int *ceed, int *vec_length,
    void (*f)(void *ctx, int *num_qpts, const CeedScalar *u, const CeedScalar *u1, const CeedScalar *u2, const CeedScalar *u3, const CeedScalar *u4,
              const CeedScalar *u5, const CeedScalar *u6, const CeedScalar *u7, const CeedScalar *u8, const CeedScalar *u9, const CeedScalar *u10,
              const CeedScalar *u11, const CeedScalar *u12, const CeedScalar *u13, const CeedScalar *u14, const CeedScalar *u15, CeedScalar *v,
              CeedScalar *v1, CeedScalar *v2, CeedScalar *v3, CeedScalar *v4, CeedScalar *v5, CeedScalar *v6, CeedScalar *v7, CeedScalar *v8,
              CeedScalar *v9, CeedScalar *v10, CeedScalar *v11, CeedScalar *v12, CeedScalar *v13, CeedScalar *v14, CeedScalar *v15, int *err),
    const char *source, int *qf, int *err, fortran_charlen_t source_len) {
  FIX_STRING(source);
  if (num_CeedQFunction == max_CeedQFunction) fCeedQFunctionExpandDict();

  CeedQFunction *qf_c = &CeedQFunction_dict[num_CeedQFunction];

  *err = CeedQFunctionCreateInterior(Ceed_dict[*ceed], *vec_length, CeedQFunctionFortranStub, source_c, qf_c);
  if (*err == 0) fCeedQFunctionAccept(qf);

  CeedQFunctionContext ctx_c;
  CeedFortranContext   ctx_data;

  *err = CeedCalloc(1, &ctx_data);
  if (*err) return;

  ctx_data->f         = f;
  ctx_data->inner_ctx = NULL;

  *err = CeedQFunctionContextCreate(Ceed_dict[*ceed], &ctx_c);
  if (*err) return;
  *err = CeedQFunctionContextSetData(ctx_c, CEED_MEM_HOST, CEED_OWN_POINTER, sizeof(*ctx_data), ctx_data);
  if (*err) return;
  *err = CeedQFunctionSetContext(*qf_c, ctx_c);
  if (*err) return;
  CeedQFunctionContextDestroy(&ctx_c);
  if (*err) return;
  *err = CeedQFunctionSetFortranStatus(*qf_c, true);
}

#define fCeedQFunctionCreateInteriorByName FORTRAN_NAME(ceedqfunctioncreateinteriorbyname, CEEDQFUNCTIONCREATEINTERIORBYNAME)
CEED_EXTERN void fCeedQFunctionCreateInteriorByName(int *ceed, const char *name, int *qf, int *err, fortran_charlen_t name_len) {
  FIX_STRING(name);
  if (num_CeedQFunction == max_CeedQFunction) fCeedQFunctionExpandDict();
  *err = CeedQFunctionCreateInteriorByName(Ceed_dict[*ceed], name_c, &CeedQFunction_dict[num_CeedQFunction]);
  if (*err == 0) fCeedQFunctionAccept(qf);
}

#define fCeedQFunctionCreateIdentity FORTRAN_NAME(ceedqfunctioncreateidentity, CEEDQFUNCTIONCREATEIDENTITY)
CEED_EXTERN void fCeedQFunctionCreateIdentity(int *ceed, int *size, int *inmode, int *outmode, int *qf, int *err) {
  if (num_CeedQFunction == max_CeedQFunction) fCeedQFunctionExpandDict();
  *err = CeedQFunctionCreateIdentity(Ceed_dict[*ceed], *size, (CeedEvalMode)*inmode, (CeedEvalMode)*outmode, &CeedQFunction_dict[num_CeedQFunction]);
  if (*err == 0) fCeedQFunctionAccept(qf);
}

#define fCeedQFunctionAddInput FORTRAN_NAME(ceedqfunctionaddinput, CEEDQFUNCTIONADDINPUT)
CEED_EXTERN void fCeedQFunctionAddInput(int *qf, const char *field_name, CeedInt *num_comp, CeedEvalMode *eval_mode, int *err,
                                        fortran_charlen_t field_name_len) {
  FIX_STRING(field_name);
  *err = CeedQFunctionAddInput(CeedQFunction_dict[*qf], field_name_c, *num_comp, *eval_mode);
}

#define fCeedQFunctionAddOutput FORTRAN_NAME(ceedqfunctionaddoutput, CEEDQFUNCTIONADDOUTPUT)
CEED_EXTERN void fCeedQFunctionAddOutput(int *qf, const char *field_name, CeedInt *num_comp, CeedEvalMode *eval_mode, int *err,
                                         fortran_charlen_t field_name_len) {
  FIX_STRING(field_name);
  *err = CeedQFunctionAddOutput(CeedQFunction_dict[*qf], field_name_c, *num_comp, *eval_mode);
}

#define fCeedQFunctionSetContext FORTRAN_NAME(ceedqfunctionsetcontext, CEEDQFUNCTIONSETCONTEXT)
CEED_EXTERN void fCeedQFunctionSetContext(int *qf, int *ctx, int *err) {
  CeedQFunctionContext ctx_c;
  CeedQFunctionContext ctx_f = CeedQFunctionContext_dict[*ctx];
  CeedFortranContext   ctx_data;

  *err = CeedQFunctionGetContext(CeedQFunction_dict[*qf], &ctx_c);
  if (*err) return;
  *err = CeedQFunctionContextGetData(ctx_c, CEED_MEM_HOST, &ctx_data);
  if (*err) return;
  ctx_data->inner_ctx = ctx_f;
  *err                = CeedQFunctionContextRestoreData(ctx_c, (void **)&ctx_data);
  if (*err) return;
  *err = CeedQFunctionContextDestroy(&ctx_c);
}

#define fCeedQFunctionSetNumViewTabs FORTRAN_NAME(ceedqfunctionsetnumviewtabs, CEEDQFUNCTIONSETNUMVIEWTABS)
CEED_EXTERN void fCeedQFunctionSetNumViewTabs(int *qf, int *num_tabs, int *err) {
  *err = CeedQFunctionSetNumViewTabs(CeedQFunction_dict[*qf], *num_tabs);
}

#define fCeedQFunctionView FORTRAN_NAME(ceedqfunctionview, CEEDQFUNCTIONVIEW)
CEED_EXTERN void fCeedQFunctionView(int *qf, int *err) { *err = CeedQFunctionView(CeedQFunction_dict[*qf], stdout); }

#define fCeedQFunctionApply FORTRAN_NAME(ceedqfunctionapply, CEEDQFUNCTIONAPPLY)
// TODO Need Fixing, double pointer
CEED_EXTERN void fCeedQFunctionApply(int *qf, int *num_qpts, int *u, int *u1, int *u2, int *u3, int *u4, int *u5, int *u6, int *u7, int *u8, int *u9,
                                     int *u10, int *u11, int *u12, int *u13, int *u14, int *u15, int *v, int *v1, int *v2, int *v3, int *v4, int *v5,
                                     int *v6, int *v7, int *v8, int *v9, int *v10, int *v11, int *v12, int *v13, int *v14, int *v15, int *err) {
  CeedVector *in;
  CeedVector *out;

  *err = CeedCalloc(CEED_FIELD_MAX, &in);
  if (*err) return;
  in[0]  = *u == FORTRAN_NULL ? NULL : CeedVector_dict[*u];
  in[1]  = *u1 == FORTRAN_NULL ? NULL : CeedVector_dict[*u1];
  in[2]  = *u2 == FORTRAN_NULL ? NULL : CeedVector_dict[*u2];
  in[3]  = *u3 == FORTRAN_NULL ? NULL : CeedVector_dict[*u3];
  in[4]  = *u4 == FORTRAN_NULL ? NULL : CeedVector_dict[*u4];
  in[5]  = *u5 == FORTRAN_NULL ? NULL : CeedVector_dict[*u5];
  in[6]  = *u6 == FORTRAN_NULL ? NULL : CeedVector_dict[*u6];
  in[7]  = *u7 == FORTRAN_NULL ? NULL : CeedVector_dict[*u7];
  in[8]  = *u8 == FORTRAN_NULL ? NULL : CeedVector_dict[*u8];
  in[9]  = *u9 == FORTRAN_NULL ? NULL : CeedVector_dict[*u9];
  in[10] = *u10 == FORTRAN_NULL ? NULL : CeedVector_dict[*u10];
  in[11] = *u11 == FORTRAN_NULL ? NULL : CeedVector_dict[*u11];
  in[12] = *u12 == FORTRAN_NULL ? NULL : CeedVector_dict[*u12];
  in[13] = *u13 == FORTRAN_NULL ? NULL : CeedVector_dict[*u13];
  in[14] = *u14 == FORTRAN_NULL ? NULL : CeedVector_dict[*u14];
  in[15] = *u15 == FORTRAN_NULL ? NULL : CeedVector_dict[*u15];

  *err = CeedCalloc(CEED_FIELD_MAX, &out);
  if (*err) return;
  out[0]  = *v == FORTRAN_NULL ? NULL : CeedVector_dict[*v];
  out[1]  = *v1 == FORTRAN_NULL ? NULL : CeedVector_dict[*v1];
  out[2]  = *v2 == FORTRAN_NULL ? NULL : CeedVector_dict[*v2];
  out[3]  = *v3 == FORTRAN_NULL ? NULL : CeedVector_dict[*v3];
  out[4]  = *v4 == FORTRAN_NULL ? NULL : CeedVector_dict[*v4];
  out[5]  = *v5 == FORTRAN_NULL ? NULL : CeedVector_dict[*v5];
  out[6]  = *v6 == FORTRAN_NULL ? NULL : CeedVector_dict[*v6];
  out[7]  = *v7 == FORTRAN_NULL ? NULL : CeedVector_dict[*v7];
  out[8]  = *v8 == FORTRAN_NULL ? NULL : CeedVector_dict[*v8];
  out[9]  = *v9 == FORTRAN_NULL ? NULL : CeedVector_dict[*v9];
  out[10] = *v10 == FORTRAN_NULL ? NULL : CeedVector_dict[*v10];
  out[11] = *v11 == FORTRAN_NULL ? NULL : CeedVector_dict[*v11];
  out[12] = *v12 == FORTRAN_NULL ? NULL : CeedVector_dict[*v12];
  out[13] = *v13 == FORTRAN_NULL ? NULL : CeedVector_dict[*v13];
  out[14] = *v14 == FORTRAN_NULL ? NULL : CeedVector_dict[*v14];
  out[15] = *v15 == FORTRAN_NULL ? NULL : CeedVector_dict[*v15];

  *err = CeedQFunctionApply(CeedQFunction_dict[*qf], *num_qpts, in, out);
  if (*err) return;

  *err = CeedFree(&in);
  if (*err) return;
  *err = CeedFree(&out);
}

#define fCeedQFunctionDestroy FORTRAN_NAME(ceedqfunctiondestroy, CEEDQFUNCTIONDESTROY)
CEED_EXTERN void fCeedQFunctionDestroy(int *qf, int *err) {
  if (*qf == FORTRAN_NULL) return;

  *err = CeedQFunctionDestroy(&CeedQFunction_dict[*qf]);
  if (*err == 0) {
    *qf = FORTRAN_NULL;
    num_active_CeedQFunction--;
    if (num_active_CeedQFunction == 0) {
      *err              = CeedFree(&CeedQFunction_dict);
      num_CeedQFunction = 0;
      max_CeedQFunction = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedOperator
// -----------------------------------------------------------------------------
static CeedOperator *CeedOperator_dict       = NULL;
static int           num_CeedOperator        = 0;
static int           num_active_CeedOperator = 0;
static int           max_CeedOperator        = 0;

static inline void fCeedOperatorExpandDict(void) {
  max_CeedOperator += max_CeedOperator / 2 + 1;
  CeedRealloc(max_CeedOperator, &CeedOperator_dict);
}

static inline void fCeedOperatorAccept(int *op) {
  *op = num_CeedOperator;
  num_CeedOperator++;
  num_active_CeedOperator++;
}

#define fCeedOperatorCreate FORTRAN_NAME(ceedoperatorcreate, CEEDOPERATORCREATE)
CEED_EXTERN void fCeedOperatorCreate(int *ceed, int *qf, int *dqf, int *dqfT, int *op, int *err) {
  if (num_CeedOperator == max_CeedOperator) fCeedOperatorExpandDict();

  CeedQFunction dqf_c = CEED_QFUNCTION_NONE, dqfT_c = CEED_QFUNCTION_NONE;
  if (*dqf != FORTRAN_QFUNCTION_NONE) dqf_c = CeedQFunction_dict[*dqf];
  if (*dqfT != FORTRAN_QFUNCTION_NONE) dqfT_c = CeedQFunction_dict[*dqfT];

  *err = CeedOperatorCreate(Ceed_dict[*ceed], CeedQFunction_dict[*qf], dqf_c, dqfT_c, &CeedOperator_dict[num_CeedOperator]);
  if (*err == 0) fCeedOperatorAccept(op);
}

#define fCeedOperatorCreateComposite FORTRAN_NAME(ceedoperatorcreatecomposite, CEEDOPERATORCREATECOMPOSITE)
CEED_EXTERN void fCeedOperatorCreateComposite(int *ceed, int *op, int *err) {
  if (num_CeedOperator == max_CeedOperator) fCeedOperatorExpandDict();
  *err = CeedOperatorCreateComposite(Ceed_dict[*ceed], &CeedOperator_dict[num_CeedOperator]);
  if (*err == 0) fCeedOperatorAccept(op);
}

#define fCeedOperatorSetField FORTRAN_NAME(ceedoperatorsetfield, CEEDOPERATORSETFIELD)
CEED_EXTERN void fCeedOperatorSetField(int *op, const char *field_name, int *rstr, int *basis, int *vec, int *err, fortran_charlen_t field_name_len) {
  FIX_STRING(field_name);
  CeedVector          vec_c;
  CeedElemRestriction rstr_c;
  CeedBasis           basis_c;

  if (*vec == FORTRAN_NULL) {
    vec_c = NULL;
  } else if (*vec == FORTRAN_VECTOR_ACTIVE) {
    vec_c = CEED_VECTOR_ACTIVE;
  } else if (*vec == FORTRAN_VECTOR_NONE) {
    vec_c = CEED_VECTOR_NONE;
  } else {
    vec_c = CeedVector_dict[*vec];
  }
  if (*rstr == FORTRAN_NULL) {
    rstr_c = NULL;
  } else if (*rstr == FORTRAN_ELEMRESTRICTION_NONE) {
    rstr_c = CEED_ELEMRESTRICTION_NONE;
  } else {
    rstr_c = CeedElemRestriction_dict[*rstr];
  }
  if (*basis == FORTRAN_NULL) {
    basis_c = NULL;
  } else if (*basis == FORTRAN_BASIS_NONE) {
    basis_c = CEED_BASIS_NONE;
  } else {
    basis_c = CeedBasis_dict[*basis];
  }

  *err = CeedOperatorSetField(CeedOperator_dict[*op], field_name_c, rstr_c, basis_c, vec_c);
}

#define fCeedOperatorCompositeAddSub FORTRAN_NAME(ceedoperatorcompositeaddsub, CEEDOPERATORCOMPOSITEADDSUB)
CEED_EXTERN void fCeedOperatorCompositeAddSub(int *composite_op, int *sub_op, int *err) {
  *err = CeedOperatorCompositeAddSub(CeedOperator_dict[*composite_op], CeedOperator_dict[*sub_op]);
}

#define fCeedOperatorSetName FORTRAN_NAME(ceedoperatorsetname, CEEDOPERATORSETNAME)
CEED_EXTERN void fCeedOperatorSetName(int *op, const char *name, int *err, fortran_charlen_t name_len) {
  FIX_STRING(name);
  *err = CeedOperatorSetName(CeedOperator_dict[*op], name_c);
}

#define fCeedOperatorSetNumViewTabs FORTRAN_NAME(ceedoperatorsetnumviewtabs, CEEDOPERATORSETNUMVIEWTABS)
CEED_EXTERN void fCeedOperatorSetNumViewTabs(int *op, int *num_tabs, int *err) {
  *err = CeedOperatorSetNumViewTabs(CeedOperator_dict[*op], *num_tabs);
}

#define fCeedOperatorLinearAssembleQFunction FORTRAN_NAME(ceedoperatorlinearassembleqfunction, CEEDOPERATORLINEARASSEMBLEQFUNCTION)
CEED_EXTERN void fCeedOperatorLinearAssembleQFunction(int *op, int *assembled_vec, int *assembled_rstr, int *request, int *err) {
  bool create_request = true;

  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*request == FORTRAN_REQUEST_IMMEDIATE || *request == FORTRAN_REQUEST_ORDERED) create_request = false;
  if (create_request && num_CeedRequest == max_CeedRequest) fCeedRequestExpandDict();

  CeedRequest *request_c;

  if (*request == FORTRAN_REQUEST_IMMEDIATE) {
    request_c = CEED_REQUEST_IMMEDIATE;
  } else if (*request == FORTRAN_REQUEST_ORDERED) {
    request_c = CEED_REQUEST_ORDERED;
  } else {
    request_c = &CeedRequest_dict[num_CeedRequest];
  }

  // Vector
  if (num_CeedVector == max_CeedVector) fCeedVectorExpandDict();

  // Restriction
  if (num_CeedElemRestriction == max_CeedElemRestriction) fCeedElemRestrictionExpandDict();

  // Assembly
  *err = CeedOperatorLinearAssembleQFunction(CeedOperator_dict[*op], &CeedVector_dict[num_CeedVector],
                                             &CeedElemRestriction_dict[num_CeedElemRestriction], request_c);
  if (*err == 0) {
    fCeedVectorAccept(assembled_vec);
    fCeedElemRestrictionAccept(assembled_rstr);
    if (create_request) fCeedRequestAccept(request);
  }
}

#define fCeedOperatorLinearAssembleDiagonal FORTRAN_NAME(ceedoperatorlinearassemblediagonal, CEEDOPERATORLINEARASSEMBLEDIAGONAL)
CEED_EXTERN void fCeedOperatorLinearAssembleDiagonal(int *op, int *assembled_vec, int *request, int *err) {
  bool create_request = true;

  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*request == FORTRAN_REQUEST_IMMEDIATE || *request == FORTRAN_REQUEST_ORDERED) create_request = false;
  if (create_request && num_CeedRequest == max_CeedRequest) fCeedRequestExpandDict();

  CeedRequest *request_c;

  if (*request == FORTRAN_REQUEST_IMMEDIATE) {
    request_c = CEED_REQUEST_IMMEDIATE;
  } else if (*request == FORTRAN_REQUEST_ORDERED) {
    request_c = CEED_REQUEST_ORDERED;
  } else {
    request_c = &CeedRequest_dict[num_CeedRequest];
  }

  *err = CeedOperatorLinearAssembleDiagonal(CeedOperator_dict[*op], CeedVector_dict[*assembled_vec], request_c);
  if (*err == 0 && create_request) fCeedRequestAccept(request);
}

#define fCeedOperatorMultigridLevelCreate FORTRAN_NAME(ceedoperatormultigridlevelcreate, CEEDOPERATORMULTIGRIDLEVELCREATE)
CEED_EXTERN void fCeedOperatorMultigridLevelCreate(int *op_fine, int *p_mult_fine, int *rstr_coarse, int *basis_coarse, int *op_coarse,
                                                   int *op_prolong, int *op_restrict, int *err) {
  // Operators
  CeedOperator op_coarse_c, op_prolong_c, op_restrict_c;

  // C interface call
  *err = CeedOperatorMultigridLevelCreate(CeedOperator_dict[*op_fine], CeedVector_dict[*p_mult_fine], CeedElemRestriction_dict[*rstr_coarse],
                                          CeedBasis_dict[*basis_coarse], &op_coarse_c, &op_prolong_c, &op_restrict_c);

  if (*err) return;
  while (num_CeedOperator + 2 >= max_CeedOperator) max_CeedOperator += max_CeedOperator / 2 + 1;
  CeedRealloc(max_CeedOperator, &CeedOperator_dict);
  CeedOperator_dict[num_CeedOperator] = op_coarse_c;
  *op_coarse                          = num_CeedOperator++;
  CeedOperator_dict[num_CeedOperator] = op_prolong_c;
  *op_prolong                         = num_CeedOperator++;
  CeedOperator_dict[num_CeedOperator] = op_restrict_c;
  *op_restrict                        = num_CeedOperator++;
  num_active_CeedOperator += 3;
}

#define fCeedOperatorMultigridLevelCreateTensorH1 FORTRAN_NAME(ceedoperatormultigridlevelcreatetensorh1, CEEDOPERATORMULTIGRIDLEVELCREATETENSORH1)
CEED_EXTERN void fCeedOperatorMultigridLevelCreateTensorH1(int *op_fine, int *p_mult_fine, int *rstr_coarse, int *basis_coarse,
                                                           const CeedScalar *interp_c_to_f, int *op_coarse, int *op_prolong, int *op_restrict,
                                                           int *err) {
  // Operators
  CeedOperator op_coarse_c, op_prolong_c, op_restrict_c;

  // C interface call
  *err = CeedOperatorMultigridLevelCreateTensorH1(CeedOperator_dict[*op_fine], CeedVector_dict[*p_mult_fine], CeedElemRestriction_dict[*rstr_coarse],
                                                  CeedBasis_dict[*basis_coarse], interp_c_to_f, &op_coarse_c, &op_prolong_c, &op_restrict_c);

  if (*err) return;
  while (num_CeedOperator + 2 >= max_CeedOperator) max_CeedOperator += max_CeedOperator / 2 + 1;
  CeedRealloc(max_CeedOperator, &CeedOperator_dict);
  CeedOperator_dict[num_CeedOperator] = op_coarse_c;
  *op_coarse                          = num_CeedOperator++;
  CeedOperator_dict[num_CeedOperator] = op_prolong_c;
  *op_prolong                         = num_CeedOperator++;
  CeedOperator_dict[num_CeedOperator] = op_restrict_c;
  *op_restrict                        = num_CeedOperator++;
  num_active_CeedOperator += 3;
}

#define fCeedOperatorMultigridLevelCreateH1 FORTRAN_NAME(ceedoperatormultigridlevelcreateh1, CEEDOPERATORMULTIGRIDLEVELCREATEH1)
CEED_EXTERN void fCeedOperatorMultigridLevelCreateH1(int *op_fine, int *p_mult_fine, int *rstr_coarse, int *basis_coarse,
                                                     const CeedScalar *interp_c_to_f, int *op_coarse, int *op_prolong, int *op_restrict, int *err) {
  // Operators
  CeedOperator op_coarse_c, op_prolong_c, op_restrict_c;

  // C interface call
  *err = CeedOperatorMultigridLevelCreateH1(CeedOperator_dict[*op_fine], CeedVector_dict[*p_mult_fine], CeedElemRestriction_dict[*rstr_coarse],
                                            CeedBasis_dict[*basis_coarse], interp_c_to_f, &op_coarse_c, &op_prolong_c, &op_restrict_c);

  if (*err) return;
  while (num_CeedOperator + 2 >= max_CeedOperator) max_CeedOperator += max_CeedOperator / 2 + 1;
  CeedRealloc(max_CeedOperator, &CeedOperator_dict);
  CeedOperator_dict[num_CeedOperator] = op_coarse_c;
  *op_coarse                          = num_CeedOperator++;
  CeedOperator_dict[num_CeedOperator] = op_prolong_c;
  *op_prolong                         = num_CeedOperator++;
  CeedOperator_dict[num_CeedOperator] = op_restrict_c;
  *op_restrict                        = num_CeedOperator++;
  num_active_CeedOperator += 3;
}

#define fCeedOperatorView FORTRAN_NAME(ceedoperatorview, CEEDOPERATORVIEW)
CEED_EXTERN void fCeedOperatorView(int *op, int *err) { *err = CeedOperatorView(CeedOperator_dict[*op], stdout); }

#define fCeedOperatorCreateFDMElementInverse FORTRAN_NAME(ceedoperatorcreatefdmelementinverse, CEEDOPERATORCREATEFDMELEMENTINVERSE)
CEED_EXTERN void fCeedOperatorCreateFDMElementInverse(int *op, int *fdm_inv, int *request, int *err) {
  bool create_request = true;

  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*request == FORTRAN_REQUEST_IMMEDIATE || *request == FORTRAN_REQUEST_ORDERED) create_request = false;
  if (create_request && num_CeedRequest == max_CeedRequest) fCeedRequestExpandDict();

  CeedRequest *request_c;

  if (*request == FORTRAN_REQUEST_IMMEDIATE) {
    request_c = CEED_REQUEST_IMMEDIATE;
  } else if (*request == FORTRAN_REQUEST_ORDERED) {
    request_c = CEED_REQUEST_ORDERED;
  } else {
    request_c = &CeedRequest_dict[num_CeedRequest];
  }

  // Operator
  if (num_CeedOperator == max_CeedOperator) fCeedOperatorExpandDict();

  *err = CeedOperatorCreateFDMElementInverse(CeedOperator_dict[*op], &CeedOperator_dict[num_CeedOperator], request_c);
  if (*err == 0) {
    fCeedOperatorAccept(fdm_inv);
    if (create_request) fCeedRequestAccept(request);
  }
}

#define fCeedOperatorApply FORTRAN_NAME(ceedoperatorapply, CEEDOPERATORAPPLY)
CEED_EXTERN void fCeedOperatorApply(int *op, int *in, int *out, int *request, int *err) {
  bool create_request = true;

  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*request == FORTRAN_REQUEST_IMMEDIATE || *request == FORTRAN_REQUEST_ORDERED) create_request = false;
  if (create_request && num_CeedRequest == max_CeedRequest) fCeedRequestExpandDict();

  CeedRequest *request_c;

  if (*request == FORTRAN_REQUEST_IMMEDIATE) {
    request_c = CEED_REQUEST_IMMEDIATE;
  } else if (*request == FORTRAN_REQUEST_ORDERED) {
    request_c = CEED_REQUEST_ORDERED;
  } else {
    request_c = &CeedRequest_dict[num_CeedRequest];
  }

  CeedVector in_c  = (*in == FORTRAN_NULL) ? NULL : (*in == FORTRAN_VECTOR_NONE ? CEED_VECTOR_NONE : CeedVector_dict[*in]);
  CeedVector out_c = (*out == FORTRAN_NULL) ? NULL : (*out == FORTRAN_VECTOR_NONE ? CEED_VECTOR_NONE : CeedVector_dict[*out]);

  *err = CeedOperatorApply(CeedOperator_dict[*op], in_c, out_c, request_c);
  if (*err == 0 && create_request) fCeedRequestAccept(request);
}

#define fCeedOperatorApplyAdd FORTRAN_NAME(ceedoperatorapplyadd, CEEDOPERATORAPPLYADD)
CEED_EXTERN void fCeedOperatorApplyAdd(int *op, int *in, int *out, int *request, int *err) {
  bool create_request = true;

  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*request == FORTRAN_REQUEST_IMMEDIATE || *request == FORTRAN_REQUEST_ORDERED) create_request = false;
  if (create_request && num_CeedRequest == max_CeedRequest) fCeedRequestExpandDict();

  CeedRequest *request_c;

  if (*request == FORTRAN_REQUEST_IMMEDIATE) {
    request_c = CEED_REQUEST_IMMEDIATE;
  } else if (*request == FORTRAN_REQUEST_ORDERED) {
    request_c = CEED_REQUEST_ORDERED;
  } else {
    request_c = &CeedRequest_dict[num_CeedRequest];
  }

  CeedVector in_c  = (*in == FORTRAN_NULL) ? NULL : (*in == FORTRAN_VECTOR_NONE ? CEED_VECTOR_NONE : CeedVector_dict[*in]);
  CeedVector out_c = (*out == FORTRAN_NULL) ? NULL : (*out == FORTRAN_VECTOR_NONE ? CEED_VECTOR_NONE : CeedVector_dict[*out]);

  *err = CeedOperatorApplyAdd(CeedOperator_dict[*op], in_c, out_c, request_c);
  if (*err == 0 && create_request) fCeedRequestAccept(request);
}

#define fCeedOperatorDestroy FORTRAN_NAME(ceedoperatordestroy, CEEDOPERATORDESTROY)
CEED_EXTERN void fCeedOperatorDestroy(int *op, int *err) {
  if (*op == FORTRAN_NULL) return;
  *err = CeedOperatorDestroy(&CeedOperator_dict[*op]);
  if (*err == 0) {
    *op = FORTRAN_NULL;
    num_active_CeedOperator--;
    if (num_active_CeedOperator == 0) {
      *err             = CeedFree(&CeedOperator_dict);
      num_CeedOperator = 0;
      max_CeedOperator = 0;
    }
  }
}

// -----------------------------------------------------------------------------
