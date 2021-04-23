// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// libCEED setup for solid mechanics example using PETSc

#include "../elasticity.h"

#include "../qfunctions/common.h"            // Geometric factors
#include "../qfunctions/Linear.h"            // Linear elasticity
#include "../qfunctions/SS-NH.h"             // Hyperelasticity small strain
// Hyperelasticity finite strain
#include "../qfunctions/FSInitial-NH1.h"     // -- Initial config 1 w/ dXref_dxinit, Grad(u) storage  / Neo-Hookean 
#include "../qfunctions/FSInitial-NH2.h"     // -- Initial config 2 w/ dXref_dxinit, Grad(u), Cinv, constant storage  / Neo-Hookean 
#include "../qfunctions/FSCurrent-NH1.h"     // -- Current config 1 w/ dXref_dxinit, Grad(u) storage  / Neo-Hookean 
#include "../qfunctions/FSCurrent-NH2.h"     // -- Current config 2 w/ dXref_dxcurr, tau, constant storage / Neo-Hookean 
#include "../qfunctions/FSInitial-MR1.h"     // -- Initial config 1 w/ dXref_dxinit, Grad(u) storage / Mooney-Rivlin
#include "../qfunctions/FSInitial-MR.h"     // -- Initial config (old version) for Neo-Hookean and Mooney-Rivlin (and start of GP)
#include "../qfunctions/constantForce.h"     // Constant forcing function
#include "../qfunctions/manufacturedForce.h" // Manufactured solution forcing
#include "../qfunctions/manufacturedTrue.h"  // Manufactured true solution

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

// -----------------------------------------------------------------------------
// Problem options
// -----------------------------------------------------------------------------
// Data specific to each problem option
problemData problem_options[10] = {
  [ELAS_LINEAR] = {
    .q_data_size = 10, // For linear elasticity, 6 would be sufficient
    .setup_geo = SetupGeo,
    .apply = ElasLinearF,
    .jacob = ElasLineardF,
    .energy = ElasLinearEnergy,
    .diagnostic = ElasLinearDiagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = ElasLinearF_loc,
    .jacob_loc = ElasLineardF_loc,
    .energy_loc = ElasLinearEnergy_loc,
    .diagnostic_loc = ElasLinearDiagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  [ELAS_SS_NH] = {
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = ElasSSNHF,
    .jacob = ElasSSNHdF,
    .energy = ElasSSNHEnergy,
    .diagnostic = ElasSSNHDiagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = ElasSSNHF_loc,
    .jacob_loc = ElasSSNHdF_loc,
    .energy_loc = ElasSSNHEnergy_loc,
    .diagnostic_loc = ElasSSNHDiagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  [ELAS_FSInitial_NH1] = {
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = ElasFSInitialNH1F,
    .jacob = ElasFSInitialNH1dF,
    .energy = ElasFSInitialNH1Energy,
    .diagnostic = ElasFSInitialNH1Diagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = ElasFSInitialNH1F_loc,
    .jacob_loc = ElasFSInitialNH1dF_loc,
    .energy_loc = ElasFSInitialNH1Energy_loc,
    .diagnostic_loc = ElasFSInitialNH1Diagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  [ELAS_FSInitial_NH2] = {
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = ElasFSInitialNH2F,
    .jacob = ElasFSInitialNH2dF,
    .energy = ElasFSInitialNH2Energy,
    .diagnostic = ElasFSInitialNH2Diagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = ElasFSInitialNH2F_loc,
    .jacob_loc = ElasFSInitialNH2dF_loc,
    .energy_loc = ElasFSInitialNH2Energy_loc,
    .diagnostic_loc = ElasFSInitialNH2Diagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  [ELAS_FSCurrent_NH1] = {
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = ElasFSCurrentNH1F,
    .jacob = ElasFSCurrentNH1dF,
    .energy = ElasFSCurrentNH1Energy,
    .diagnostic = ElasFSCurrentNH1Diagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = ElasFSCurrentNH1F_loc,
    .jacob_loc = ElasFSCurrentNH1dF_loc,
    .energy_loc = ElasFSCurrentNH1Energy_loc,
    .diagnostic_loc = ElasFSCurrentNH1Diagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  [ELAS_FSCurrent_NH2] = {
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = ElasFSCurrentNH2F,
    .jacob = ElasFSCurrentNH2dF,
    .energy = ElasFSCurrentNH2Energy,
    .diagnostic = ElasFSCurrentNH2Diagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = ElasFSCurrentNH2F_loc,
    .jacob_loc = ElasFSCurrentNH2dF_loc,
    .energy_loc = ElasFSCurrentNH2Energy_loc,
    .diagnostic_loc = ElasFSCurrentNH2Diagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  [ELAS_FSInitial_MR1] = {
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = ElasFSInitialMR1F,
    .jacob = ElasFSInitialMR1dF,
    .energy = ElasFSInitialMR1Energy,
    .diagnostic = ElasFSInitialMR1Diagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = ElasFSInitialMR1F_loc,
    .jacob_loc = ElasFSInitialMR1dF_loc,
    .energy_loc = ElasFSInitialMR1Energy_loc,
    .diagnostic_loc = ElasFSInitialMR1Diagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  //Neo-Hookean - old version
  [ELAS_HYPER_FS_NH] = { 
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = HyperFSF_NH,
    .jacob = HyperFSdF_NH,
    .energy = HyperFSEnergy_NH,
    .diagnostic = HyperFSDiagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = HyperFSF_NH_loc,
    .jacob_loc = HyperFSdF_NH_loc,
    .energy_loc = HyperFSEnergy_NH_loc,
    .diagnostic_loc = HyperFSDiagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  //Mooney-Rivlin
  [ELAS_HYPER_FS_MR] = { 
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = HyperFSF_MR,
    .jacob = HyperFSdF_MR,
    .energy = HyperFSEnergy_MR,
    .diagnostic = HyperFSDiagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = HyperFSF_MR_loc,
    .jacob_loc = HyperFSdF_MR_loc,
    .energy_loc = HyperFSEnergy_MR_loc,
    .diagnostic_loc = HyperFSDiagnostic_loc,
    .quad_mode = CEED_GAUSS
  },
  //Generalized Polynomial
  [ELAS_HYPER_FS_GP] = { 
    .q_data_size = 10,
    .setup_geo = SetupGeo,
    .apply = HyperFSF_GP,
    .jacob = HyperFSdF_GP,
    .energy = HyperFSEnergy_GP,
    .diagnostic = HyperFSDiagnostic,
    .setup_geo_loc = SetupGeo_loc,
    .apply_loc = HyperFSF_GP_loc,
    .jacob_loc = HyperFSdF_GP_loc,
    .energy_loc = HyperFSEnergy_GP_loc,
    .diagnostic_loc = HyperFSDiagnostic_loc,
    .quad_mode = CEED_GAUSS
  }
};

// Forcing function data
forcingData forcing_options[3] = {
  [FORCE_NONE] = {
    .setup_forcing = NULL,
    .setup_forcing_loc = NULL
  },
  [FORCE_CONST] = {
    .setup_forcing = SetupConstantForce,
    .setup_forcing_loc = SetupConstantForce_loc
  },
  [FORCE_MMS] = {
    .setup_forcing = SetupMMSForce,
    .setup_forcing_loc = SetupMMSForce_loc
  }
};

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedInt level, CeedData data) {
  PetscErrorCode ierr;

  // Vectors
  CeedVectorDestroy(&data->q_data);
  CeedVectorDestroy(&data->q_data_diagnostic);
  CeedVectorDestroy(&data->grad_u);
  CeedVectorDestroy(&data->C_inv);
  CeedVectorDestroy(&data->lam_log_J);
  CeedVectorDestroy(&data->dXdx);
  CeedVectorDestroy(&data->tau);
  CeedVectorDestroy(&data->x_ceed);
  CeedVectorDestroy(&data->y_ceed);
  CeedVectorDestroy(&data->true_soln);

  // Restrictions
  CeedElemRestrictionDestroy(&data->elem_restr_u);
  CeedElemRestrictionDestroy(&data->elem_restr_x);
  CeedElemRestrictionDestroy(&data->elem_restr_gradu_i);
  CeedElemRestrictionDestroy(&data->elem_restr_C_inv);
  CeedElemRestrictionDestroy(&data->elem_restr_lam_log_J);
  CeedElemRestrictionDestroy(&data->elem_restr_dXdx);
  CeedElemRestrictionDestroy(&data->elem_restr_tau);
  CeedElemRestrictionDestroy(&data->elem_restr_qd_i);
  CeedElemRestrictionDestroy(&data->elem_restr_energy);
  CeedElemRestrictionDestroy(&data->elem_restr_diagnostic);
  CeedElemRestrictionDestroy(&data->elem_restr_qd_diagnostic_i);

  // Bases
  CeedBasisDestroy(&data->basis_x);
  CeedBasisDestroy(&data->basis_u);
  CeedBasisDestroy(&data->basis_energy);
  CeedBasisDestroy(&data->basis_diagnostic);

  // QFunctions
  CeedQFunctionDestroy(&data->qf_jacob);
  CeedQFunctionDestroy(&data->qf_apply);
  CeedQFunctionDestroy(&data->qf_energy);
  CeedQFunctionDestroy(&data->qf_diagnostic);

  // Operators
  CeedOperatorDestroy(&data->op_jacob);
  CeedOperatorDestroy(&data->op_apply);
  CeedOperatorDestroy(&data->op_energy);
  CeedOperatorDestroy(&data->op_diagnostic);

  // Restriction and Prolongation data
  CeedBasisDestroy(&data->basis_c_to_f);
  CeedOperatorDestroy(&data->op_prolong);
  CeedOperatorDestroy(&data->op_restrict);

  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i + 1);
};

// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt height, DMLabel domain_label, CeedInt value,
    CeedElemRestriction *elem_restr) {

  PetscSection section;
  PetscInt p, num_elem, num_dof, *restr_indices, elem_offset, num_fields, dim,
           depth;
  DMLabel depth_label;
  IS depth_is, iter_is;
  Vec U_loc;
  const PetscInt *iter_indices;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_offsets[num_fields+1];
  field_offsets[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_offsets[f+1] = field_offsets[f] + num_comp[f];
  }

  ierr = DMPlexGetDepth(dm, &depth); CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depth_label); CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depth_label, depth - height, &depth_is);
  CHKERRQ(ierr);
  if (domain_label) {
    IS domain_is;
    ierr = DMLabelGetStratumIS(domain_label, value, &domain_is); CHKERRQ(ierr);
    if (domain_is) { // domainIS is non-empty
      ierr = ISIntersect(depth_is, domain_is, &iter_is); CHKERRQ(ierr);
      ierr = ISDestroy(&domain_is); CHKERRQ(ierr);
    } else { // domainIS is NULL (empty)
      iter_is = NULL;
    }
    ierr = ISDestroy(&depth_is); CHKERRQ(ierr);
  } else {
    iter_is = depth_is;
  }
  if (iter_is) {
    ierr = ISGetLocalSize(iter_is, &num_elem); CHKERRQ(ierr);
    ierr = ISGetIndices(iter_is, &iter_indices); CHKERRQ(ierr);
  } else {
    num_elem = 0;
    iter_indices = NULL;
  }
  ierr = PetscMalloc1(num_elem*PetscPowInt(P, dim), &restr_indices);
  CHKERRQ(ierr);
  for (p = 0, elem_offset = 0; p < num_elem; p++) {
    PetscInt c = iter_indices[p];
    PetscInt num_indices, *indices, num_nodes;
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE,
                                   &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    bool flip = false;
    if (height > 0) {
      PetscInt num_cells, num_faces, start = -1;
      const PetscInt *orients, *faces, *cells;
      ierr = DMPlexGetSupport(dm, c, &cells); CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, c, &num_cells); CHKERRQ(ierr);
      if (num_cells != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                                     "Expected one cell in support of exterior face, but got %D cells",
                                     num_cells);
      ierr = DMPlexGetCone(dm, cells[0], &faces); CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, cells[0], &num_faces); CHKERRQ(ierr);
      for (PetscInt i=0; i<num_faces; i++) {if (faces[i] == c) start = i;}
      if (start < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_CORRUPT,
                                "Could not find face %D in cone of its support",
                                c);
      ierr = DMPlexGetConeOrientation(dm, cells[0], &orients); CHKERRQ(ierr);
      if (orients[start] < 0) flip = true;
    }
    if (num_indices % field_offsets[num_fields]) SETERRQ1(PETSC_COMM_SELF,
          PETSC_ERR_ARG_INCOMP, "Number of closure indices not compatible with Cell %D",
          c);
    num_nodes = num_indices / field_offsets[num_fields];
    for (PetscInt i = 0; i < num_nodes; i++) {
      PetscInt ii = i;
      if (flip) {
        if (P == num_nodes) ii = num_nodes - 1 - i;
        else if (P*P == num_nodes) {
          PetscInt row = i / P, col = i % P;
          ii = row + col * P;
        } else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP,
                          "No support for flipping point with %D nodes != P (%D) or P^2",
                          num_nodes, P);
      }
      // Check that indices are blocked by node and thus can be coalesced as a single field with
      // field_offsets[num_fields] = sum(num_comp) components.
      for (PetscInt f = 0; f < num_fields; f++) {
        for (PetscInt j = 0; j < num_comp[f]; j++) {
          if (Involute(indices[field_offsets[f]*num_nodes + ii*num_comp[f] + j])
              != Involute(indices[ii*num_comp[0]]) + field_offsets[f] + j)
            SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                     "Cell %D closure indices not interlaced for node %D field %D component %D",
                     c, ii, f, j);
        }
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = Involute(indices[ii*num_comp[0]]);
      restr_indices[elem_offset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (elem_offset != num_elem*PetscPowInt(P, dim))
    SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
             "ElemRestriction of size (%D,%D) initialized %D nodes", num_elem,
             PetscPowInt(P, dim),elem_offset);
  if (iter_is) {
    ierr = ISRestoreIndices(iter_is, &iter_indices); CHKERRQ(ierr);
  }
  ierr = ISDestroy(&iter_is); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U_loc, &num_dof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &U_loc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, num_elem, PetscPowInt(P, dim),
                            field_offsets[num_fields],
                            1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices,
                            elem_restr);
  ierr = PetscFree(restr_indices); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height,
                                       DMLabel domain_label, PetscInt value,
                                       CeedInt P, CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q,
                                       CeedElemRestriction *elem_restr_x,
                                       CeedElemRestriction *elem_restr_qd_i) {

  DM dm_coord;
  CeedInt dim, num_local_elem;
  CeedInt Q_dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  dim -= height;
  Q_dim = CeedIntPow(Q, dim);
  ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  if (elem_restr_q) {
    ierr = CreateRestrictionFromPlex(ceed, dm, P, height, domain_label, value,
                                     elem_restr_q); CHKERRQ(ierr);
  }
  if (elem_restr_x) {
    ierr = CreateRestrictionFromPlex(ceed, dm_coord, 2, height, domain_label,
                                     value, elem_restr_x); CHKERRQ(ierr);
  }
  if (elem_restr_qd_i) {
    CeedElemRestrictionGetNumElements(*elem_restr_q, &num_local_elem);
    CeedElemRestrictionCreateStrided(ceed, num_local_elem, Q_dim,
                                     q_data_size, q_data_size*num_local_elem*Q_dim,
                                     CEED_STRIDES_BACKEND, elem_restr_qd_i);
  }

  PetscFunctionReturn(0);
};

// Set up libCEED on the fine grid for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, DM dm_energy, DM dm_diagnostic,
                                     Ceed ceed, AppCtx app_ctx,
                                     CeedQFunctionContext phys_ctx,
                                     CeedData *data, PetscInt fine_level,
                                     PetscInt num_comp_u, PetscInt U_g_size,
                                     PetscInt U_loc_size, CeedVector force_ceed,
                                     CeedVector neumann_ceed) {
  int           ierr;
  CeedInt       P = app_ctx->level_degrees[fine_level] + 1;
  CeedInt       Q = app_ctx->level_degrees[fine_level] + 1 + app_ctx->q_extra;
  CeedInt       dim, num_comp_x, num_comp_e = 1, num_comp_d = 5;
  CeedInt       num_qpts;
  CeedInt       q_data_size =
    problem_options[app_ctx->problem_choice].q_data_size;
  problemType   problem_choice = app_ctx->problem_choice;
  forcingType   forcing_choice = app_ctx->forcing_choice;
  DM            dm_coord;
  Vec           coords;
  PetscInt      c_start, c_end, num_elem;
  const PetscScalar *coordArray;
  CeedVector    x_coord;
  CeedQFunction qf_setup_geo, qf_apply, qf_jacob, qf_energy, qf_diagnostic;
  CeedOperator  op_setup_geo, op_apply, op_jacob, op_energy, op_diagnostic;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  num_comp_x = dim;

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // -- Coordinate restriction
  ierr = CreateRestrictionFromPlex(ceed, dm_coord, 2, 0, 0, 0,
                                   &(data[fine_level]->elem_restr_x));
  CHKERRQ(ierr);
  // -- Solution restriction
  ierr = CreateRestrictionFromPlex(ceed, dm, P, 0, 0, 0,
                                   &data[fine_level]->elem_restr_u);
  CHKERRQ(ierr);
  // -- Energy restriction
  ierr = CreateRestrictionFromPlex(ceed, dm_energy, P, 0, 0, 0,
                                   &data[fine_level]->elem_restr_energy);
  CHKERRQ(ierr);
  // -- Pressure restriction
  ierr = CreateRestrictionFromPlex(ceed, dm_diagnostic, P, 0, 0, 0,
                                   &data[fine_level]->elem_restr_diagnostic);
  CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;

  // -- Geometric data restriction
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, q_data_size,
                                   q_data_size*num_elem*Q*Q*Q,
                                   CEED_STRIDES_BACKEND,
                                   &data[fine_level]->elem_restr_qd_i);
  // -- State vector gradient restriction
  switch (problem_choice) {
  // ---- Linear Elasticity
  case ELAS_LINEAR:
    break;
  // ---- Hyperelasticity at small strain
  case ELAS_SS_NH:
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    break;
  // ---- Hyperelasticity at finite strain
  case ELAS_FSInitial_NH1:
    // ------ Storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    break;
  case ELAS_FSInitial_NH2:
    // ------ Storage: dXdx, Grad(u), C_inv, lamda*logJ
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, (dim+1)*num_comp_u/2,
                                     (dim+1)*num_comp_u*num_elem*Q*Q*Q/2,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_C_inv);
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, 1,
                                     1*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_lam_log_J);
    break;
  case ELAS_FSCurrent_NH1:
    // ------ Storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    break;
  case ELAS_FSCurrent_NH2:
    // ------ Storage: dXdxcur, tau, mu - lamda*logJ
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_dXdx);

    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, (dim+1)*num_comp_u/2,
                                     (dim+1)*num_comp_u*num_elem*Q*Q*Q/2,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_tau);
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, 1,
                                     1*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_lam_log_J);
    break;
  case ELAS_FSInitial_MR1:
    // ------ Storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    break;
  case ELAS_HYPER_FS_NH:
    // ------ Storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    break;
  case ELAS_HYPER_FS_MR:
    // ------ Storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    break;
  case ELAS_HYPER_FS_GP:
    // ------ Storage: dXdx, Grad(u)
    CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q*Q, dim*num_comp_u,
                                     dim*num_comp_u*num_elem*Q*Q*Q,
                                     CEED_STRIDES_BACKEND,
                                     &data[fine_level]->elem_restr_gradu_i);
    break;
  }
  // -- Geometric data restriction
  CeedElemRestrictionCreateStrided(ceed, num_elem, P*P*P, q_data_size,
                                   q_data_size*num_elem*P*P*P,
                                   CEED_STRIDES_BACKEND,
                                   &data[fine_level]->elem_restr_qd_diagnostic_i);

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(data[fine_level]->elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // libCEED bases
  // ---------------------------------------------------------------------------
  // -- Solution basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, P, Q,
                                  problem_options[problem_choice].quad_mode,
                                  &data[fine_level]->basis_u);
  // -- Coordinate basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q,
                                  problem_options[problem_choice].quad_mode,
                                  &data[fine_level]->basis_x);
  // -- Energy basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_e, P, Q,
                                  problem_options[problem_choice].quad_mode,
                                  &data[fine_level]->basis_energy);
  // -- Diagnostic output basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, P, P, CEED_GAUSS_LOBATTO,
                                  &data[fine_level]->basis_diagnostic);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  CeedBasisGetNumQuadraturePoints(data[fine_level]->basis_u, &num_qpts);
  // -- Geometric data vector
  CeedVectorCreate(ceed, q_data_size*num_elem*num_qpts,
                   &data[fine_level]->q_data);
  // -- Collocated geometric data vector
  CeedVectorCreate(ceed, q_data_size*num_elem*P*P*P,
                   &data[fine_level]->q_data_diagnostic);
  // -- State gradient vector
  switch (problem_choice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    break;
  case ELAS_FSInitial_NH1:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    break;
  case ELAS_FSInitial_NH2:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    CeedVectorCreate(ceed, (dim+1)*num_comp_u*num_elem*num_qpts/2,
                     &data[fine_level]->C_inv);
    CeedVectorCreate(ceed, 1*num_elem*num_qpts, &data[fine_level]->lam_log_J);
    break;
  case ELAS_FSCurrent_NH1:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    break;
  case ELAS_FSCurrent_NH2:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->dXdx);
    CeedVectorCreate(ceed, (dim+1)*num_comp_u*num_elem*num_qpts/2,
                     &data[fine_level]->tau);
    CeedVectorCreate(ceed, 1*num_elem*num_qpts, &data[fine_level]->lam_log_J);
    break;
  case ELAS_FSInitial_MR1:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_NH:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_MR:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_GP:
    CeedVectorCreate(ceed, dim*num_comp_u*num_elem*num_qpts,
                     &data[fine_level]->grad_u);
    break;
  }
  // -- Operator action variables
  CeedVectorCreate(ceed, U_loc_size, &data[fine_level]->x_ceed);
  CeedVectorCreate(ceed, U_loc_size, &data[fine_level]->y_ceed);

  // ---------------------------------------------------------------------------
  // Geometric factor computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the quadrature data
  //   q_data returns dXdx_i,j and w * det.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_options[problem_choice].setup_geo,
                              problem_options[problem_choice].setup_geo_loc,
                              &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "q_data", q_data_size, CEED_EVAL_NONE);

  // -- Operator
  CeedOperatorCreate(ceed, qf_setup_geo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "dx", data[fine_level]->elem_restr_x,
                       data[fine_level]->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE,
                       data[fine_level]->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "q_data", data[fine_level]->elem_restr_qd_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Compute the quadrature data
  CeedOperatorApply(op_setup_geo, x_coord, data[fine_level]->q_data,
                    CEED_REQUEST_IMMEDIATE);

  // -- Cleanup
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);

  // ---------------------------------------------------------------------------
  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the
  //   non-linear PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_options[problem_choice].apply,
                              problem_options[problem_choice].apply_loc,
                              &qf_apply);
  CeedQFunctionAddInput(qf_apply, "du", num_comp_u*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_apply, "q_data", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "dv", num_comp_u*dim, CEED_EVAL_GRAD);
  switch (problem_choice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH1:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH2:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_apply, "C_inv", num_comp_u*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_apply, "lam_log_J", 1, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH1:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH2:
    CeedQFunctionAddOutput(qf_apply, "dXdx", num_comp_u*dim, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_apply, "tau", num_comp_u*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_apply, "lam_log_J", 1, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_MR1:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_HYPER_FS_NH:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_HYPER_FS_MR:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_HYPER_FS_GP:
    CeedQFunctionAddOutput(qf_apply, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  }
  CeedQFunctionSetContext(qf_apply, phys_ctx);

  // -- Operator
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_apply);
  CeedOperatorSetField(op_apply, "du", data[fine_level]->elem_restr_u,
                       data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "q_data", data[fine_level]->elem_restr_qd_i,
                       CEED_BASIS_COLLOCATED, data[fine_level]->q_data);
  CeedOperatorSetField(op_apply, "dv", data[fine_level]->elem_restr_u,
                       data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  switch (problem_choice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_FSInitial_NH1:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_FSInitial_NH2:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    CeedOperatorSetField(op_apply, "C_inv", data[fine_level]->elem_restr_C_inv,
                         CEED_BASIS_COLLOCATED, data[fine_level]->C_inv);
    CeedOperatorSetField(op_apply, "lam_log_J",
                         data[fine_level]->elem_restr_lam_log_J,
                         CEED_BASIS_COLLOCATED, data[fine_level]->lam_log_J);
    break;
  case ELAS_FSCurrent_NH1:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_FSCurrent_NH2:
    CeedOperatorSetField(op_apply, "dXdx", data[fine_level]->elem_restr_dXdx,
                         CEED_BASIS_COLLOCATED, data[fine_level]->dXdx);
    CeedOperatorSetField(op_apply, "tau", data[fine_level]->elem_restr_tau,
                         CEED_BASIS_COLLOCATED, data[fine_level]->tau);
    CeedOperatorSetField(op_apply, "lam_log_J",
                         data[fine_level]->elem_restr_lam_log_J,
                         CEED_BASIS_COLLOCATED, data[fine_level]->lam_log_J);
    break;
  case ELAS_FSInitial_MR1:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_NH:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_MR:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_GP:
    CeedOperatorSetField(op_apply, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  }
  // -- Save libCEED data
  data[fine_level]->qf_apply = qf_apply;
  data[fine_level]->op_apply = op_apply;

  // ---------------------------------------------------------------------------
  // Jacobian evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the action of the
  //   Jacobian for each linear solve.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_options[problem_choice].jacob,
                              problem_options[problem_choice].jacob_loc,
                              &qf_jacob);
  CeedQFunctionAddInput(qf_jacob, "deltadu", num_comp_u*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_jacob, "q_data", q_data_size, CEED_EVAL_NONE);
  switch (problem_choice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH1:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_NH2:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_jacob, "C_inv", num_comp_u*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_jacob, "lam_log_J", 1, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH1:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_FSCurrent_NH2:
    CeedQFunctionAddInput(qf_jacob, "dXdx", num_comp_u*dim, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_jacob, "tau", num_comp_u*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_jacob, "lam_log_J", 1, CEED_EVAL_NONE);
    break;
  case ELAS_FSInitial_MR1:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_HYPER_FS_NH:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_HYPER_FS_MR:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  case ELAS_HYPER_FS_GP:
    CeedQFunctionAddInput(qf_jacob, "gradu", num_comp_u*dim, CEED_EVAL_NONE);
    break;
  }
  CeedQFunctionAddOutput(qf_jacob, "deltadv", num_comp_u*dim, CEED_EVAL_GRAD);
  CeedQFunctionSetContext(qf_jacob, phys_ctx);

  // -- Operator
  CeedOperatorCreate(ceed, qf_jacob, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_jacob);
  CeedOperatorSetField(op_jacob, "deltadu", data[fine_level]->elem_restr_u,
                       data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_jacob, "q_data", data[fine_level]->elem_restr_qd_i,
                       CEED_BASIS_COLLOCATED, data[fine_level]->q_data);
  CeedOperatorSetField(op_jacob, "deltadv", data[fine_level]->elem_restr_u,
                       data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  switch (problem_choice) {
  case ELAS_LINEAR:
    break;
  case ELAS_SS_NH:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_FSInitial_NH1:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_FSInitial_NH2:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    CeedOperatorSetField(op_jacob, "C_inv", data[fine_level]->elem_restr_C_inv,
                         CEED_BASIS_COLLOCATED, data[fine_level]->C_inv);
    CeedOperatorSetField(op_jacob, "lam_log_J",
                         data[fine_level]->elem_restr_lam_log_J,
                         CEED_BASIS_COLLOCATED, data[fine_level]->lam_log_J);
    break;
  case ELAS_FSCurrent_NH1:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_FSCurrent_NH2:
    CeedOperatorSetField(op_jacob, "dXdx", data[fine_level]->elem_restr_dXdx,
                         CEED_BASIS_COLLOCATED, data[fine_level]->dXdx);
    CeedOperatorSetField(op_jacob, "tau", data[fine_level]->elem_restr_tau,
                         CEED_BASIS_COLLOCATED, data[fine_level]->tau);
    CeedOperatorSetField(op_jacob, "lam_log_J",
                         data[fine_level]->elem_restr_lam_log_J,
                         CEED_BASIS_COLLOCATED, data[fine_level]->lam_log_J);
    break;
  case ELAS_FSInitial_MR1:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_NH:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_MR:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  case ELAS_HYPER_FS_GP:
    CeedOperatorSetField(op_jacob, "gradu", data[fine_level]->elem_restr_gradu_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->grad_u);
    break;
  }
  // -- Save libCEED data
  data[fine_level]->qf_jacob = qf_jacob;
  data[fine_level]->op_jacob = op_jacob;

  // ---------------------------------------------------------------------------
  // Traction boundary conditions, if needed
  // ---------------------------------------------------------------------------
  if (app_ctx->bc_traction_count > 0) {
    // -- Setup
    DMLabel domain_label;
    ierr = DMGetLabel(dm, "Face Sets", &domain_label); CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

    // -- Basis
    CeedInt height = 1;
    CeedBasis basis_u_face, basis_x_face;
    CeedBasisCreateTensorH1Lagrange(ceed, dim - height, num_comp_u, P, Q,
                                    problem_options[problem_choice].quad_mode,
                                    &basis_u_face);
    CeedBasisCreateTensorH1Lagrange(ceed, dim - height, num_comp_x, 2, Q,
                                    problem_options[problem_choice].quad_mode,
                                    &basis_x_face);

    // -- QFunction
    CeedQFunction qf_traction;
    CeedQFunctionContext traction_ctx;
    CeedQFunctionCreateInterior(ceed, 1, SetupTractionBCs, SetupTractionBCs_loc,
                                &qf_traction);
    CeedQFunctionContextCreate(ceed, &traction_ctx);
    CeedQFunctionSetContext(qf_traction, traction_ctx);
    CeedQFunctionAddInput(qf_traction, "dx", num_comp_x*(num_comp_x - height),
                          CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_traction, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qf_traction, "v", num_comp_u, CEED_EVAL_INTERP);

    // -- Compute contribution on each boundary face
    for (CeedInt i = 0; i < app_ctx->bc_traction_count; i++) {
      CeedElemRestriction elem_restr_u_face, elem_restr_x_face;
      CeedOperator op_traction;
      CeedQFunctionContextSetData(traction_ctx, CEED_MEM_HOST, CEED_USE_POINTER,
                                  3 * sizeof(CeedScalar),
                                  app_ctx->bc_traction_vector[i]);

      // Setup restriction
      ierr = GetRestrictionForDomain(ceed, dm, height, domain_label,
                                     app_ctx->bc_traction_faces[i], P, Q,
                                     0, &elem_restr_u_face, &elem_restr_x_face, NULL);
      CHKERRQ(ierr);

      // ---- Create boundary Operator
      CeedOperatorCreate(ceed, qf_traction, NULL, NULL, &op_traction);
      CeedOperatorSetField(op_traction, "dx", elem_restr_x_face, basis_x_face,
                           CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_traction, "weight", CEED_ELEMRESTRICTION_NONE,
                           basis_x_face, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_traction, "v", elem_restr_u_face,
                           basis_u_face, CEED_VECTOR_ACTIVE);

      // ---- Compute traction on face
      CeedOperatorApplyAdd(op_traction, x_coord, neumann_ceed,
                           CEED_REQUEST_IMMEDIATE);

      // ---- Cleanup
      CeedElemRestrictionDestroy(&elem_restr_u_face);
      CeedElemRestrictionDestroy(&elem_restr_x_face);
      CeedOperatorDestroy(&op_traction);
    }

    // -- Cleanup
    CeedBasisDestroy(&basis_u_face);
    CeedBasisDestroy(&basis_x_face);
    CeedQFunctionDestroy(&qf_traction);
    CeedQFunctionContextDestroy(&traction_ctx);
  }

  // ---------------------------------------------------------------------------
  // Forcing term, if needed
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the forcing term (RHS)
  //   for the non-linear PDE.
  // ---------------------------------------------------------------------------
  if (forcing_choice != FORCE_NONE) {
    CeedQFunction qf_setup_force;
    CeedOperator op_setup_force;

    // -- QFunction
    CeedQFunctionCreateInterior(ceed, 1,
                                forcing_options[forcing_choice].setup_forcing,
                                forcing_options[forcing_choice].setup_forcing_loc,
                                &qf_setup_force);
    CeedQFunctionAddInput(qf_setup_force, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setup_force, "q_data", q_data_size, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_force, "force", num_comp_u, CEED_EVAL_INTERP);
    if (forcing_choice == FORCE_MMS) {
      CeedQFunctionSetContext(qf_setup_force, phys_ctx);
    } else {
      CeedQFunctionContext ctxForcing;
      CeedQFunctionContextCreate(ceed, &ctxForcing);
      CeedQFunctionContextSetData(ctxForcing, CEED_MEM_HOST, CEED_USE_POINTER,
                                  sizeof(*app_ctx->forcing_vector),
                                  app_ctx->forcing_vector);
      CeedQFunctionSetContext(qf_setup_force, ctxForcing);
      CeedQFunctionContextDestroy(&ctxForcing);
    }

    // -- Operator
    CeedOperatorCreate(ceed, qf_setup_force, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &op_setup_force);
    CeedOperatorSetField(op_setup_force, "x", data[fine_level]->elem_restr_x,
                         data[fine_level]->basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_force, "q_data",
                         data[fine_level]->elem_restr_qd_i,
                         CEED_BASIS_COLLOCATED, data[fine_level]->q_data);
    CeedOperatorSetField(op_setup_force, "force", data[fine_level]->elem_restr_u,
                         data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);

    // -- Compute forcing term
    CeedOperatorApply(op_setup_force, x_coord, force_ceed, CEED_REQUEST_IMMEDIATE);

    // -- Cleanup
    CeedQFunctionDestroy(&qf_setup_force);
    CeedOperatorDestroy(&op_setup_force);
  }

  // ---------------------------------------------------------------------------
  // True solution, for MMS
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the true solution at
  //   the mesh nodes for validation with the manufactured solution.
  // ---------------------------------------------------------------------------
  if (forcing_choice == FORCE_MMS) {
    CeedScalar *true_array;
    const CeedScalar *mult_array;
    CeedVector mult_vec;
    CeedBasis basis_x_true;
    CeedQFunction qf_true;
    CeedOperator op_true;

    // -- Solution vector
    CeedVectorCreate(ceed, U_loc_size, &(data[fine_level]->true_soln));

    // -- Basis
    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, P, CEED_GAUSS_LOBATTO,
                                    &basis_x_true);

    // QFunction
    CeedQFunctionCreateInterior(ceed, 1, MMSTrueSoln, MMSTrueSoln_loc,
                                &qf_true);
    CeedQFunctionAddInput(qf_true, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_true, "true_soln", num_comp_u, CEED_EVAL_NONE);

    // Operator
    CeedOperatorCreate(ceed, qf_true, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                       &op_true);
    CeedOperatorSetField(op_true, "x", data[fine_level]->elem_restr_x, basis_x_true,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_true, "true_soln", data[fine_level]->elem_restr_u,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    // -- Compute true solution
    CeedOperatorApply(op_true, x_coord, data[fine_level]->true_soln,
                      CEED_REQUEST_IMMEDIATE);

    // -- Multiplicity calculation
    CeedElemRestrictionCreateVector(data[fine_level]->elem_restr_u, &mult_vec,
                                    NULL);
    CeedVectorSetValue(mult_vec, 0.);
    CeedElemRestrictionGetMultiplicity(data[fine_level]->elem_restr_u, mult_vec);

    // -- Multiplicity correction
    CeedVectorGetArray(data[fine_level]->true_soln, CEED_MEM_HOST, &true_array);
    CeedVectorGetArrayRead(mult_vec, CEED_MEM_HOST, &mult_array);
    for (CeedInt i = 0; i < U_loc_size; i++)
      true_array[i] /= mult_array[i];
    CeedVectorRestoreArray(data[fine_level]->true_soln, &true_array);
    CeedVectorRestoreArrayRead(mult_vec, &mult_array);

    // -- Cleanup
    CeedVectorDestroy(&mult_vec);
    CeedBasisDestroy(&basis_x_true);
    CeedQFunctionDestroy(&qf_true);
    CeedOperatorDestroy(&op_true);
  }

  // ---------------------------------------------------------------------------
  // Local energy computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the strain energy
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_options[problem_choice].energy,
                              problem_options[problem_choice].energy_loc,
                              &qf_energy);
  CeedQFunctionAddInput(qf_energy, "du", num_comp_u*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_energy, "q_data", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_energy, "energy", num_comp_e, CEED_EVAL_INTERP);
  CeedQFunctionSetContext(qf_energy, phys_ctx);

  // -- Operator
  CeedOperatorCreate(ceed, qf_energy, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_energy);
  CeedOperatorSetField(op_energy, "du", data[fine_level]->elem_restr_u,
                       data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_energy, "q_data", data[fine_level]->elem_restr_qd_i,
                       CEED_BASIS_COLLOCATED, data[fine_level]->q_data);
  CeedOperatorSetField(op_energy, "energy", data[fine_level]->elem_restr_energy,
                       data[fine_level]->basis_energy, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data
  data[fine_level]->qf_energy = qf_energy;
  data[fine_level]->op_energy = op_energy;

  // ---------------------------------------------------------------------------
  // Diagnostic value computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes nodal diagnostic quantities
  // ---------------------------------------------------------------------------
  // Geometric factors
  // -- Coordinate basis
  CeedBasis basis_x;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, CEED_GAUSS_LOBATTO,
                                  &basis_x);
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_options[problem_choice].setup_geo,
                              problem_options[problem_choice].setup_geo_loc,
                              &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "q_data", q_data_size, CEED_EVAL_NONE);

  // -- Operator
  CeedOperatorCreate(ceed, qf_setup_geo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "dx", data[fine_level]->elem_restr_x,
                       basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE,
                       basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "q_data",
                       data[fine_level]->elem_restr_qd_diagnostic_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // -- Compute the quadrature data
  CeedOperatorApply(op_setup_geo, x_coord, data[fine_level]->q_data_diagnostic,
                    CEED_REQUEST_IMMEDIATE);

  // -- Cleanup
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);

  // Diagnostic quantities
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_options[problem_choice].diagnostic,
                              problem_options[problem_choice].diagnostic_loc,
                              &qf_diagnostic);
  CeedQFunctionAddInput(qf_diagnostic, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_diagnostic, "du", num_comp_u*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diagnostic, "q_data", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diagnostic, "diagnostic", num_comp_u + num_comp_d,
                         CEED_EVAL_NONE);
  CeedQFunctionSetContext(qf_diagnostic, phys_ctx);

  // -- Operator
  CeedOperatorCreate(ceed, qf_diagnostic, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_diagnostic);
  CeedOperatorSetField(op_diagnostic, "u", data[fine_level]->elem_restr_u,
                       data[fine_level]->basis_diagnostic, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diagnostic, "du", data[fine_level]->elem_restr_u,
                       data[fine_level]->basis_diagnostic, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diagnostic, "q_data",
                       data[fine_level]->elem_restr_qd_diagnostic_i,
                       CEED_BASIS_COLLOCATED, data[fine_level]->q_data_diagnostic);
  CeedOperatorSetField(op_diagnostic, "diagnostic",
                       data[fine_level]->elem_restr_diagnostic,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data
  data[fine_level]->qf_diagnostic = qf_diagnostic;
  data[fine_level]->op_diagnostic = op_diagnostic;

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  CeedVectorDestroy(&x_coord);

  PetscFunctionReturn(0);
};

// Set up libCEED multigrid level for a given degree
//   Prolongation and Restriction are between level and level+1
PetscErrorCode SetupLibceedLevel(DM dm, Ceed ceed, AppCtx app_ctx,
                                 CeedData *data, PetscInt level,
                                 PetscInt num_comp_u, PetscInt U_g_size,
                                 PetscInt U_loc_size, CeedVector fine_mult) {
  PetscErrorCode ierr;
  CeedInt        fine_level = app_ctx->num_levels - 1;
  CeedInt        P = app_ctx->level_degrees[level] + 1;
  CeedInt        Q = app_ctx->level_degrees[fine_level] + 1 + app_ctx->q_extra;
  CeedInt        dim;
  CeedOperator   op_jacob, op_prolong, op_restrict;

  PetscFunctionBeginUser;

  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  // -- Solution restriction
  ierr = CreateRestrictionFromPlex(ceed, dm, P, 0, 0, 0,
                                   &data[level]->elem_restr_u);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // libCEED bases
  // ---------------------------------------------------------------------------
  // -- Solution basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, P, Q,
                                  problem_options[app_ctx->problem_choice].quad_mode,
                                  &data[level]->basis_u);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  CeedVectorCreate(ceed, U_loc_size, &data[level]->x_ceed);
  CeedVectorCreate(ceed, U_loc_size, &data[level]->y_ceed);

  // ---------------------------------------------------------------------------
  // Coarse Grid, Prolongation, and Restriction Operators
  // ---------------------------------------------------------------------------
  // Create the Operators that compute the prolongation and
  //   restriction between the p-multigrid levels and the coarse grid eval.
  // ---------------------------------------------------------------------------
  CeedOperatorMultigridLevelCreate(data[level+1]->op_jacob, fine_mult,
                                   data[level]->elem_restr_u, data[level]->basis_u,
                                   &op_jacob, &op_prolong, &op_restrict);

  // -- Save libCEED data
  data[level]->op_jacob = op_jacob;
  data[level+1]->op_prolong = op_prolong;
  data[level+1]->op_restrict = op_restrict;

  PetscFunctionReturn(0);
};
