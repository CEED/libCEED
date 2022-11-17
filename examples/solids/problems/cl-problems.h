// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef cl_problems_h
#define cl_problems_h

// Problem options
typedef enum {
  ELAS_LINEAR        = 0,
  ELAS_SS_NH         = 1,
  ELAS_FSInitial_NH1 = 2,
  ELAS_FSInitial_NH2 = 3,
  ELAS_FSCurrent_NH1 = 4,
  ELAS_FSCurrent_NH2 = 5,
  ELAS_FSInitial_MR1 = 6
} problemType;
static const char *const problemTypes[]        = {"Linear",        "SS-NH",         "FSInitial-NH1", "FSInitial-NH2", "FSCurrent-NH1",
                                                  "FSCurrent-NH2", "FSInitial-MR1", "problemType",   "ELAS_",         0};
static const char *const problemTypesForDisp[] = {
    "Linear elasticity",
    "Hyperelasticity small strain, Neo-Hookean",
    "Hyperelasticity finite strain Initial configuration Neo-Hookean w/ dXref_dxinit, Grad(u) storage",
    "Hyperelasticity finite strain Initial configuration Neo-Hookean w/ dXref_dxinit, Grad(u), C_inv, constant storage",
    "Hyperelasticity finite strain Current configuration Neo-Hookean w/ dXref_dxinit, Grad(u) storage",
    "Hyperelasticity finite strain Current configuration Neo-Hookean w/ dXref_dxcurr, tau, constant storage",
    "Hyperelasticity finite strain Initial configuration Moony-Rivlin w/ dXref_dxinit, Grad(u) storage"};

#endif  // cl_problems_h