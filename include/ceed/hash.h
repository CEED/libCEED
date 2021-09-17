/// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
/// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
/// reserved. See files LICENSE and NOTICE for details.
///
/// This file is part of CEED, a collection of benchmarks, miniapps, software
/// libraries and APIs for efficient high-order finite element and spectral
/// element discretizations for exascale applications. For more information and
/// source code availability see http://github.com/ceed.
///
/// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
/// a collaborative effort of two U.S. Department of Energy organizations (Office
/// of Science and the National Nuclear Security Administration) responsible for
/// the planning and preparation of a capable exascale ecosystem, including
/// software, applications, hardware, advanced system engineering and early
/// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Public header for hashing functionality for libCEED, adapted from PETSc
#ifndef _ceed_hash_h
#define _ceed_hash_h

#include <ceed/ceed.h>
#include <ceed/khash.h>

/* Required for khash <= 0.2.5 */
#if !defined(kcalloc)
#define kcalloc(N,Z) calloc(N,Z)
#endif
#if !defined(kmalloc)
#define kmalloc(Z) malloc(Z)
#endif
#if !defined(krealloc)
#define krealloc(P,Z) realloc(P,Z)
#endif
#if !defined(kfree)
#define kfree(P) free(P)
#endif

#define CeedHashGetValue(ht,k,v) ((v) = kh_value((ht),(k)))

#define CeedHashMissing(ht,k) ((k) == kh_end((ht)))

/* --- Thomas Wang integer hash functions --- */

typedef khint32_t CeedHash32_t;
typedef khint64_t CeedHash64_t;
typedef khint_t   CeedHash_t;

/* Thomas Wang's second version for 32bit integers */
static inline CeedHash_t CeedHash_UInt32(CeedHash32_t key) {
  key = ~key + (key << 15); /* key = (key << 15) - key - 1; */
  key =  key ^ (key >> 12);
  key =  key + (key <<  2);
  key =  key ^ (key >>  4);
  key =  key * 2057;        /* key = (key + (key << 3)) + (key << 11); */
  key =  key ^ (key >> 16);
  return key;
}

static inline CeedHash_t CeedHashInt(CeedInt key) {
  return CeedHash_UInt32((CeedHash32_t)key);
}

static inline CeedHash_t CeedHashCombine(CeedHash_t seed, CeedHash_t hash) {
  /* https://doi.org/10.1002/asi.10170 */
  /* https://dl.acm.org/citation.cfm?id=759509 */
  return seed ^ (hash + (seed << 6) + (seed >> 2));
}

// IJ - two keys
typedef struct _CeedHashIJKey { CeedInt i, j; } CeedHashIJKey;
#define CeedHashIJKeyHash(key) \
  CeedHashCombine(CeedHashInt((key).i),CeedHashInt((key).j))

#define CeedHashIJKeyEqual(k1,k2) \
  ((k1).i==(k2).i && (k1).j==(k2).j)

#define CeedHashIJInit(name, value)										\
	KHASH_INIT(name,CeedHashIJKey,value,1,CeedHashIJKeyHash,CeedHashIJKeyEqual)

typedef struct _CeedHashIJKKey { CeedInt i, j, k; } CeedHashIJKKey;
#define CeedHashIJKKeyHash(key) \
  CeedHashCombine(CeedHashCombine(CeedHashInt((key).i),CeedHashInt((key).j)), \
                  CeedHashInt((key).k))

// IJK - three keys
#define CeedHashIJKKeyEqual(k1,k2) \
  ((k1).i==(k2).i && (k1).j==(k2).j && (k1).k==(k2).k)

#define CeedHashIJKInit(name, value)										\
	KHASH_INIT(name,CeedHashIJKKey,value,1,CeedHashIJKKeyHash,CeedHashIJKKeyEqual)

// IJKL - four keys
typedef struct _CeedHashIJKLKey { CeedInt i, j, k, l; } CeedHashIJKLKey;
#define CeedHashIJKLKeyHash(key) \
  CeedHashCombine(CeedHashCombine(CeedHashInt((key).i),CeedHashInt((key).j)), \
                  CeedHashCombine(CeedHashInt((key).k),CeedHashInt((key).l)))

#define CeedHashIJKLKeyEqual(k1,k2) \
  ((k1).i==(k2).i && (k1).j==(k2).j && (k1).k==(k2).k && (k1).l==(k2).l)

#define CeedHashIJKLInit(name, value)										\
	KHASH_INIT(name,CeedHashIJKLKey,value,1,CeedHashIJKLKeyHash,CeedHashIJKLKeyEqual)

// IJKLM - five keys
typedef struct _CeedHashIJKLMKey { CeedInt i, j, k, l, m; } CeedHashIJKLMKey;
#define CeedHashIJKLMKeyHash(key) \
  CeedHashCombine( \
  CeedHashCombine(CeedHashCombine(CeedHashInt((key).i),CeedHashInt((key).j)), \
                  CeedHashCombine(CeedHashInt((key).k),CeedHashInt((key).l))), \
                  CeedHashInt((key).m))

#define CeedHashIJKLMKeyEqual(k1,k2) \
  ((k1).i==(k2).i && (k1).j==(k2).j && (k1).k==(k2).k && (k1).l==(k2).l && (k1).m==(k2).m)

#define CeedHashIJKLMInit(name, value)										\
	KHASH_INIT(name,CeedHashIJKLMKey,value,1,CeedHashIJKLMKeyHash,CeedHashIJKLMKeyEqual)

#endif // _ceed_hash_h
