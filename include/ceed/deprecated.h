/// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for user and utility components of libCEED
#pragma once

#if __STDC_VERSION__ >= 202311L
#define DEPRECATED(msg) [[deprecated(msg)]]
#elif defined(__GNUC__) || defined(__clang__)
#define DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define DEPRECATED(msg)
#endif

// Compatibility with previous composite CeedOperator naming
DEPRECATED("Use CeedOperatorCreateComposite()")
static inline int CeedCompositeOperatorCreate(Ceed a, CeedOperator *b) { return CeedOperatorCreateComposite(a, b); }
DEPRECATED("Use CeedOperatorCompositeAddSub()")
static inline int CeedCompositeOperatorAddSub(CeedOperator a, CeedOperator b) { return CeedOperatorCompositeAddSub(a, b); }
DEPRECATED("Use CeedOperatorCompositeGetNumSub()")
static inline int CeedCompositeOperatorGetNumSub(CeedOperator a, CeedInt *b) { return CeedOperatorCompositeGetNumSub(a, b); }
DEPRECATED("Use CeedOperatorCompositeGetSubList()")
static inline int CeedCompositeOperatorGetSubList(CeedOperator a, CeedOperator **b) { return CeedOperatorCompositeGetSubList(a, b); }
DEPRECATED("Use CeedOperatorCompositeGetSubByName()")
static inline int CeedCompositeOperatorGetSubByName(CeedOperator a, const char *b, CeedOperator *c) {
  return CeedOperatorCompositeGetSubByName(a, b, c);
}
DEPRECATED("Use CeedOperatorCompositeGetMultiplicity()")
static inline int CeedCompositeOperatorGetMultiplicity(CeedOperator a, CeedInt b, CeedInt *c, CeedVector d) {
  return CeedOperatorCompositeGetMultiplicity(a, b, c, d);
}

#undef DEPRECATED
