// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef stabilization_types_h
#define stabilization_types_h

typedef enum {
  STAB_NONE = 0,
  STAB_SU   = 1,  // Streamline Upwind
  STAB_SUPG = 2,  // Streamline Upwind Petrov-Galerkin
} StabilizationType;

#endif  // stabilization_types_h
