// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up statistics collection

#include <petscsf.h>

#include "../navierstokes.h"
#include "petscsys.h"

PetscErrorCode CreateStatsDM(User user, ProblemData *problem, PetscInt degree, SimpleBC bc) {
  user->spanstats.num_comp_stats = 1;
  PetscFunctionBeginUser;

  // Get DM from surface
  {
    DMLabel label;
    PetscCall(DMGetLabel(user->dm, "Face Sets", &label));
    PetscCall(DMPlexLabelComplete(user->dm, label));
    PetscCall(DMPlexFilter(user->dm, label, 1, &user->spanstats.dm));
    PetscCall(DMProjectCoordinates(user->spanstats.dm, NULL));  // Ensure that a coordinate FE exists
  }

  PetscCall(PetscObjectSetName((PetscObject)user->spanstats.dm, "Spanwise_Stats"));
  PetscCall(DMSetOptionsPrefix(user->spanstats.dm, "spanstats_"));
  PetscCall(PetscOptionsSetValue(NULL, "-spanstats_dm_sparse_localize", "0"));  // [Jed] Not relevant because not periodic in this direction

  PetscCall(DMSetFromOptions(user->spanstats.dm));
  PetscCall(DMViewFromOptions(user->spanstats.dm, NULL, "-dm_view"));  // -spanstats_dm_view (option includes prefix)
  {
    PetscFE fe;
    DMLabel label;

    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, problem->dim - 1, user->spanstats.num_comp_stats, PETSC_FALSE, degree, PETSC_DECIDE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "stats"));
    PetscCall(DMAddField(user->spanstats.dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(user->spanstats.dm));
    PetscCall(DMGetLabel(user->spanstats.dm, "Face Sets", &label));

    // // Set wall BCs
    // if (bc->num_wall > 0) {
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, bc->num_wall, bc->walls, 0, bc->num_comps, bc->wall_comps,
    //                           (void (*)(void))problem->bc, NULL, problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the x direction
    // if (bc->num_slip[0] > 0) {
    //   PetscInt comps[1] = {1};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipx", label, bc->num_slip[0], bc->slips[0], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the y direction
    // if (bc->num_slip[1] > 0) {
    //   PetscInt comps[1] = {2};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipy", label, bc->num_slip[1], bc->slips[1], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }
    // // Set slip BCs in the z direction
    // if (bc->num_slip[2] > 0) {
    //   PetscInt comps[1] = {3};
    //   PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "slipz", label, bc->num_slip[2], bc->slips[2], 0, 1, comps, (void (*)(void))NULL, NULL,
    //                           problem->bc_ctx, NULL));
    // }

    PetscCall(DMPlexSetClosurePermutationTensor(user->spanstats.dm, PETSC_DETERMINE, NULL));
    PetscCall(PetscFEDestroy(&fe));
  }

  PetscSection section;
  PetscCall(DMGetLocalSection(user->spanstats.dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "Test"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 0, "Mean Velocity Products XX"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 1, "Mean Velocity Products YY"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 2, "Mean Velocity Products ZZ"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 3, "Mean Velocity Products YZ"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 4, "Mean Velocity Products XZ"));
  // PetscCall(PetscSectionSetComponentName(section, 0, 5, "Mean Velocity Products XY"));

  Vec test;
  PetscCall(DMCreateLocalVector(user->spanstats.dm, &test));
  PetscCall(VecZeroEntries(test));
  PetscCall(VecViewFromOptions(test, NULL, "-test_view"));

  PetscFunctionReturn(0);
}

PetscErrorCode GetQuadratureCoords(Ceed ceed, DM dm, CeedElemRestriction elem_restr_x, CeedBasis basis_x, CeedVector x_coords, CeedVector *qx_coords,
                                   PetscInt *total_nqpnts) {
  CeedQFunction       qf_quad_coords;
  CeedOperator        op_quad_coords;
  PetscInt            num_comp_x, loc_num_elem, num_elem_qpts;
  CeedElemRestriction elem_restr_qx;
  PetscFunctionBeginUser;

  // Create Element Restriction and CeedVector for quadrature coordinates
  CeedBasisGetNumQuadraturePoints(basis_x, &num_elem_qpts);
  CeedElemRestrictionGetNumElements(elem_restr_x, &loc_num_elem);
  CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x);
  *total_nqpnts           = num_elem_qpts * loc_num_elem;
  const CeedInt strides[] = {num_comp_x, 1, num_elem_qpts * num_comp_x};
  CeedElemRestrictionCreateStrided(ceed, loc_num_elem, num_elem_qpts, num_comp_x, num_comp_x * loc_num_elem * num_elem_qpts, strides, &elem_restr_qx);
  CeedElemRestrictionCreateVector(elem_restr_qx, qx_coords, NULL);

  // Create QFunction
  CeedQFunctionCreateIdentity(ceed, num_comp_x, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_quad_coords);

  // Create Operator
  CeedOperatorCreate(ceed, qf_quad_coords, NULL, NULL, &op_quad_coords);
  CeedOperatorSetField(op_quad_coords, "input", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_quad_coords, "output", elem_restr_qx, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorApply(op_quad_coords, x_coords, *qx_coords, CEED_REQUEST_IMMEDIATE);

  CeedQFunctionDestroy(&qf_quad_coords);
  CeedOperatorDestroy(&op_quad_coords);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateStatsSF(Ceed ceed, CeedData ceed_data, DM parentdm, DM childdm, PetscSF statssf) {
  PetscInt   child_num_qpnts, parent_num_qpnts, num_comp_x;
  CeedVector child_qx_coords, parent_qx_coords;
  PetscReal *child_coords, *parent_coords;
  PetscFunctionBeginUser;

  // Assume that child and parent have the same number of components
  CeedBasisGetNumComponents(ceed_data->basis_x, &num_comp_x);
  const PetscInt num_comp_sf = num_comp_x - 1;  // Number of coord components used in the creation of the SF

  // Get quad_coords for child DM
  PetscCall(GetQuadratureCoords(ceed, childdm, ceed_data->elem_restr_x, ceed_data->basis_xc, ceed_data->x_coord, &child_qx_coords, &child_num_qpnts));

  // Get quad_coords for parent DM
  PetscCall(GetQuadratureCoords(ceed, parentdm, ceed_data->spanstats.elem_restr_parent_x, ceed_data->spanstats.basis_x, ceed_data->spanstats.x_coord,
                                &parent_qx_coords, &parent_num_qpnts));

  // Remove z component of coordinates for matching
  {
    const PetscReal *child_quad_coords, *parent_quad_coords;

    CeedVectorGetArrayRead(child_qx_coords, CEED_MEM_HOST, &child_quad_coords);
    CeedVectorGetArrayRead(parent_qx_coords, CEED_MEM_HOST, &parent_quad_coords);

    PetscCall(PetscMalloc2(child_num_qpnts * 2, &child_coords, parent_num_qpnts * 2, &parent_coords));
    for (int i = 0; i < child_num_qpnts; i++) {
      child_coords[0 + i * num_comp_sf] = child_quad_coords[0 + i * num_comp_x];
      child_coords[1 + i * num_comp_sf] = child_quad_coords[1 + i * num_comp_x];
    }
    for (int i = 0; i < parent_num_qpnts; i++) {
      parent_coords[0 + i * num_comp_sf] = parent_quad_coords[0 + i * num_comp_x];
      parent_coords[1 + i * num_comp_sf] = parent_quad_coords[1 + i * num_comp_x];
    }
    CeedVectorRestoreArrayRead(child_qx_coords, &child_quad_coords);
    CeedVectorRestoreArrayRead(parent_qx_coords, &parent_quad_coords);
  }

  // Only check the first two components of the coordinates
  PetscCall(PetscSFSetGraphFromCoordinates(statssf, parent_num_qpnts, child_num_qpnts, num_comp_sf, 1e-12, parent_coords, child_coords));

  PetscCall(PetscFree2(child_coords, parent_coords));
  CeedVectorDestroy(&ceed_data->spanstats.x_coord);
  CeedVectorDestroy(&child_qx_coords);
  CeedVectorDestroy(&parent_qx_coords);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupStatsCollection(Ceed ceed, User user, CeedData ceed_data, ProblemData *problem) {
  DM                 dm   = user->spanstats.dm;
  MPI_Comm           comm = PetscObjectComm((PetscObject)dm);
  CeedInt            dim, P, Q, num_comp_x;
  Vec                X_loc;
  PetscMemType       X_loc_memtype;
  const PetscScalar *X_loc_array;
  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(dm, &dim));
  CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_x, &Q);
  CeedBasisGetNumNodes1D(ceed_data->basis_x, &P);

  // TODO: Possibly need to create a elem_restr_qcollocated for the global domain as well
  PetscCall(GetRestrictionForDomain(ceed, dm, 0, 0, 0, Q, user->spanstats.num_comp_stats, &ceed_data->spanstats.elem_restr_parent_stats,
                                    &ceed_data->spanstats.elem_restr_parent_x, &ceed_data->spanstats.elem_restr_parent_qd));
  CeedElemRestrictionGetNumComponents(ceed_data->spanstats.elem_restr_parent_x, &num_comp_x);
  CeedElemRestrictionCreateVector(ceed_data->spanstats.elem_restr_parent_x, &ceed_data->spanstats.x_coord, NULL);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, CEED_GAUSS_LOBATTO, &ceed_data->spanstats.basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, user->spanstats.num_comp_stats, P, Q, CEED_GAUSS, &ceed_data->spanstats.basis_stats);

  // -- Copy DM coordinates into CeedVector
  {
    DM cdm;
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (cdm) {
      PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
    } else {
      PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
    }
  }
  PetscCall(VecScale(X_loc, problem->dm_scale));
  PetscCall(VecGetArrayReadAndMemType(X_loc, &X_loc_array, &X_loc_memtype));
  CeedVectorSetArray(ceed_data->spanstats.x_coord, MemTypeP2C(X_loc_memtype), CEED_COPY_VALUES, (PetscScalar *)X_loc_array);
  PetscCall(VecRestoreArrayRead(X_loc, &X_loc_array));

  // Create SF for communicating child data back their respective parents
  PetscCall(PetscSFCreate(comm, &user->spanstats.sf));
  PetscCall(CreateStatsSF(ceed, ceed_data, user->dm, user->spanstats.dm, user->spanstats.sf));

  PetscFunctionReturn(0);
}
