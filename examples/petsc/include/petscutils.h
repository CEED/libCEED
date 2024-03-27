// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for using PETSc with libCEED
#pragma once

#include <ceed.h>
#include <petsc.h>

#include "structs.h"
#if PETSC_VERSION_LT(3, 21, 0)
#define DMSetCoordinateDisc(a, b, c) DMProjectCoordinates(a, b)
#endif

CeedMemType      MemTypeP2C(PetscMemType mtype);
PetscErrorCode   VecP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode   VecC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode   VecReadP2C(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed);
PetscErrorCode   VecReadC2P(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc);
PetscErrorCode   Kershaw(DM dm_orig, PetscScalar eps);
PetscErrorCode   SetupDMByDegree(DM dm, PetscInt p_degree, PetscInt q_extra, PetscInt num_comp_u, PetscInt topo_dim, bool enforce_bc);
PetscErrorCode   CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type);
PetscErrorCode   DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field);
PetscErrorCode   BasisCreateFromTabulation(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt face, PetscFE fe,
                                           PetscTabulation basis_tabulation, PetscQuadrature quadrature, CeedBasis *basis);
PetscErrorCode   CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field, BPData bp_data,
                                     CeedBasis *basis);
PetscErrorCode   CreateDistributedDM(RunParams rp, DM *dm);
