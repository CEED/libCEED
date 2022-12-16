#include "../include/setup-fe.h"

#include "petscerror.h"

// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

// ---------------------------------------------------------------------------
// Setup FE
// ---------------------------------------------------------------------------
PetscErrorCode SetupFEHdiv(MPI_Comm comm, DM dm, DM dm_u0, DM dm_p0) {
  PetscSection sec, sec_u0, sec_p0;
  PetscInt     dofs_per_face;
  PetscInt     p_start, p_end;
  PetscInt     c_start, c_end;  // cells
  PetscInt     f_start, f_end;  // faces
  PetscInt     v_start, v_end;  // vertices

  PetscFunctionBeginUser;

  // Get plex limits
  PetscCall(DMPlexGetChart(dm, &p_start, &p_end));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &f_start, &f_end));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &v_start, &v_end));
  // Create section for coupled problem
  PetscCall(PetscSectionCreate(comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, 2));
  PetscCall(PetscSectionSetFieldName(sec, 0, "Velocity"));
  PetscCall(PetscSectionSetFieldComponents(sec, 0, 1));
  PetscCall(PetscSectionSetFieldName(sec, 1, "Pressure"));
  PetscCall(PetscSectionSetFieldComponents(sec, 1, 1));
  PetscCall(PetscSectionSetChart(sec, p_start, p_end));
  // Create section for initial conditions u0
  PetscCall(PetscSectionCreate(comm, &sec_u0));
  PetscCall(PetscSectionSetNumFields(sec_u0, 1));
  PetscCall(PetscSectionSetFieldName(sec_u0, 0, "Velocity"));
  PetscCall(PetscSectionSetFieldComponents(sec_u0, 0, 1));
  PetscCall(PetscSectionSetChart(sec_u0, p_start, p_end));
  // Create section for initial conditions p0
  PetscCall(PetscSectionCreate(comm, &sec_p0));
  PetscCall(PetscSectionSetNumFields(sec_p0, 1));
  PetscCall(PetscSectionSetFieldName(sec_p0, 0, "Pressure"));
  PetscCall(PetscSectionSetFieldComponents(sec_p0, 0, 1));
  PetscCall(PetscSectionSetChart(sec_p0, p_start, p_end));
  // Setup dofs per face for velocity field
  for (PetscInt f = f_start; f < f_end; f++) {
    PetscCall(DMPlexGetConeSize(dm, f, &dofs_per_face));
    PetscCall(PetscSectionSetFieldDof(sec, f, 0, dofs_per_face));
    PetscCall(PetscSectionSetDof(sec, f, dofs_per_face));

    PetscCall(DMPlexGetConeSize(dm_u0, f, &dofs_per_face));
    PetscCall(PetscSectionSetFieldDof(sec_u0, f, 0, dofs_per_face));
    PetscCall(PetscSectionSetDof(sec_u0, f, dofs_per_face));
  }
  // Setup 1 dof per cell for pressure field
  for (PetscInt c = c_start; c < c_end; c++) {
    PetscCall(PetscSectionSetFieldDof(sec, c, 1, 1));
    PetscCall(PetscSectionSetDof(sec, c, 1));

    PetscCall(PetscSectionSetFieldDof(sec_p0, c, 0, 1));
    PetscCall(PetscSectionSetDof(sec_p0, c, 1));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetSection(dm, sec));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscSectionDestroy(&sec));
  PetscCall(PetscSectionSetUp(sec_u0));
  PetscCall(DMSetSection(dm_u0, sec_u0));
  PetscCall(DMCreateDS(dm_u0));
  PetscCall(PetscSectionDestroy(&sec_u0));
  PetscCall(PetscSectionSetUp(sec_p0));
  PetscCall(DMSetSection(dm_p0, sec_p0));
  PetscCall(DMCreateDS(dm_p0));
  PetscCall(PetscSectionDestroy(&sec_p0));

  PetscFunctionReturn(0);
};

// ---------------------------------------------------------------------------
// Set-up FE for H1 space
// ---------------------------------------------------------------------------
PetscErrorCode SetupFEH1(ProblemData problem_data, AppCtx app_ctx, DM dm_H1) {
  // Two FE space for displacement and pressure
  PetscFE fe;
  // number of quadrature points
  PetscInt  q_degree   = app_ctx->degree + 2 + app_ctx->q_extra;
  PetscBool is_simplex = PETSC_TRUE;
  PetscFunctionBeginUser;

  // Check if simplex or tensor-product element
  PetscCall(DMPlexIsSimplex(dm_H1, &is_simplex));
  // Create FE space
  PetscCall(PetscFECreateLagrange(app_ctx->comm, problem_data->dim, problem_data->dim, is_simplex, app_ctx->degree, q_degree, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "U"));
  PetscCall(DMAddField(dm_H1, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm_H1));

  {
    // create FE field for coordinates
    //  PetscFE fe_coords;
    //  PetscInt num_comp_coord;
    //  PetscCall( DMGetCoordinateDim(dm_H1, &num_comp_coord) );
    //  PetscCall( PetscFECreateLagrange(app_ctx->comm, problem_data->dim,
    //                                   num_comp_coord,
    //                                   is_simplex, 1, q_degree,
    //                                   &fe_coords) );
    //  PetscCall( DMProjectCoordinates(dm_H1, fe_coords) );
    //  PetscCall( PetscFEDestroy(&fe_coords) );
  }
  PetscCall(DMPlexSetClosurePermutationTensor(dm_H1, PETSC_DETERMINE, NULL));
  // Cleanup
  PetscCall(PetscFEDestroy(&fe));

  // Empty name for conserved field (because there is only one field)
  PetscSection section;
  PetscCall(DMGetLocalSection(dm_H1, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, "Velocity"));
  if (problem_data->dim == 2) {
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "Velocity_X"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "Velocity_Y"));
  } else {
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "Velocity_X"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "Velocity_Y"));
    PetscCall(PetscSectionSetComponentName(section, 0, 2, "Velocity_Z"));
  }
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Utility function - convert from DMPolytopeType to CeedElemTopology
// -----------------------------------------------------------------------------
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type) {
  switch (cell_type) {
    case DM_POLYTOPE_TRIANGLE:
      return CEED_TOPOLOGY_TRIANGLE;
    case DM_POLYTOPE_QUADRILATERAL:
      return CEED_TOPOLOGY_QUAD;
    case DM_POLYTOPE_TETRAHEDRON:
      return CEED_TOPOLOGY_TET;
    case DM_POLYTOPE_HEXAHEDRON:
      return CEED_TOPOLOGY_HEX;
    default:
      return 0;
  }
};

// -----------------------------------------------------------------------------
// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
// -----------------------------------------------------------------------------
PetscInt Involute(PetscInt i) { return i >= 0 ? i : -(i + 1); };

// -----------------------------------------------------------------------------
// Get CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets;

  PetscFunctionBeginUser;

  PetscCall(DMPlexGetLocalOffsets(dm, domain_label, value, height, 0, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets));

  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, elem_restr_offsets, elem_restr);
  PetscCall(PetscFree(elem_restr_offsets));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Get Oriented CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm, DM dm_u0, DM dm_p0, CeedInt P, CeedElemRestriction *elem_restr_u,
                                                 CeedElemRestriction *elem_restr_p, CeedElemRestriction *elem_restr_u0,
                                                 CeedElemRestriction *elem_restr_p0) {
  PetscSection section, section_u0, section_p0;
  PetscInt     p, num_elem, num_dof, num_dof_u0, num_dof_p0, *restr_indices_u, *restr_indices_p, *restr_indices_u0, *restr_indices_p0, elem_offset,
      num_fields, num_fields_u0, num_fields_p0, dim, c_start, c_end;
  Vec             U_loc;
  const PetscInt *ornt;  // this is for orientation of dof
  PetscFunctionBeginUser;
  // Section for mixed problem
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &num_fields));
  PetscInt num_comp[num_fields], field_offsets[num_fields + 1];
  field_offsets[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    PetscCall(PetscSectionGetFieldComponents(section, f, &num_comp[f]));
    field_offsets[f + 1] = field_offsets[f] + num_comp[f];
  }
  // Section for initial conditions u0
  PetscCall(DMGetLocalSection(dm_u0, &section_u0));
  PetscCall(PetscSectionGetNumFields(section_u0, &num_fields_u0));
  PetscInt num_comp_u0[num_fields_u0], field_offsets_u0[num_fields_u0 + 1];
  field_offsets_u0[0] = 0;
  for (PetscInt f = 0; f < num_fields_u0; f++) {
    PetscCall(PetscSectionGetFieldComponents(section_u0, f, &num_comp_u0[f]));
    field_offsets_u0[f + 1] = field_offsets_u0[f] + num_comp_u0[f];
  }
  // Section for initial conditions p0
  PetscCall(DMGetLocalSection(dm_p0, &section_p0));
  PetscCall(PetscSectionGetNumFields(section_p0, &num_fields_p0));
  PetscInt num_comp_p0[num_fields_p0], field_offsets_p0[num_fields_p0 + 1];
  field_offsets_p0[0] = 0;
  for (PetscInt f = 0; f < num_fields_p0; f++) {
    PetscCall(PetscSectionGetFieldComponents(section_p0, f, &num_comp_p0[f]));
    field_offsets_p0[f + 1] = field_offsets_p0[f] + num_comp_p0[f];
  }

  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  PetscCall(PetscMalloc1(num_elem * dim * PetscPowInt(P, dim), &restr_indices_u));
  PetscCall(PetscMalloc1(num_elem * dim * PetscPowInt(P, dim), &restr_indices_u0));
  PetscCall(PetscMalloc1(num_elem, &restr_indices_p));
  PetscCall(PetscMalloc1(num_elem, &restr_indices_p0));
  bool *orient_indices_u, *orient_indices_u0;  // to flip the dof
  PetscCall(PetscMalloc1(num_elem * dim * PetscPowInt(P, dim), &orient_indices_u));
  PetscCall(PetscMalloc1(num_elem * dim * PetscPowInt(P, dim), &orient_indices_u0));
  for (p = 0, elem_offset = 0; p < num_elem; p++) {
    PetscInt num_indices, *indices, faces_per_elem, dofs_per_face, num_indices_u0, *indices_u0, num_indices_p0, *indices_p0;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, p, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCall(DMPlexGetClosureIndices(dm_u0, section_u0, section_u0, p, PETSC_TRUE, &num_indices_u0, &indices_u0, NULL, NULL));
    PetscCall(DMPlexGetClosureIndices(dm_p0, section_p0, section_p0, p, PETSC_TRUE, &num_indices_p0, &indices_p0, NULL, NULL));
    restr_indices_p[p]  = indices[num_indices - 1];
    restr_indices_p0[p] = indices_p0[0];
    PetscCall(DMPlexGetConeOrientation(dm, p, &ornt));
    // Get number of faces per element
    PetscCall(DMPlexGetConeSize(dm, p, &faces_per_elem));
    dofs_per_face = faces_per_elem - 2;
    for (PetscInt f = 0; f < faces_per_elem; f++) {
      for (PetscInt i = 0; i < dofs_per_face; i++) {
        PetscInt ii = dofs_per_face * f + i;
        // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
        PetscInt loc                 = Involute(indices[ii * num_comp[0]]);
        restr_indices_u[elem_offset] = loc;
        // Set orientation
        orient_indices_u[elem_offset] = ornt[f] < 0;
        PetscInt loc_u0               = Involute(indices_u0[ii * num_comp_u0[0]]);
        restr_indices_u0[elem_offset] = loc_u0;
        // Set orientation
        orient_indices_u0[elem_offset] = ornt[f] < 0;
        elem_offset++;
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, p, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(dm_u0, section_u0, section_u0, p, PETSC_TRUE, &num_indices_u0, &indices_u0, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(dm_p0, section_p0, section_p0, p, PETSC_TRUE, &num_indices_p0, &indices_p0, NULL, NULL));
  }
  // if (elem_offset != num_elem*dim*PetscPowInt(P, dim))
  //   SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
  //           "ElemRestriction of size (%" PetscInt_FMT ", %" PetscInt_FMT" )
  //           initialized %" PetscInt_FMT " nodes", num_elem,
  //           dim*PetscPowInt(P, dim),elem_offset);

  PetscCall(DMGetLocalVector(dm, &U_loc));
  PetscCall(VecGetLocalSize(U_loc, &num_dof));
  PetscCall(DMRestoreLocalVector(dm, &U_loc));
  // dof per element in Hdiv is dim*P^dim, for linear element P=2
  CeedElemRestrictionCreateOriented(ceed, num_elem, dim * PetscPowInt(P, dim), 1, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices_u,
                                    orient_indices_u, elem_restr_u);
  CeedElemRestrictionCreate(ceed, num_elem, 1, 1, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices_p, elem_restr_p);
  PetscCall(DMGetLocalVector(dm_u0, &U_loc));
  PetscCall(VecGetLocalSize(U_loc, &num_dof_u0));
  PetscCall(DMRestoreLocalVector(dm_u0, &U_loc));
  // dof per element in Hdiv is dim*P^dim, for linear element P=2
  CeedElemRestrictionCreateOriented(ceed, num_elem, dim * PetscPowInt(P, dim), 1, 1, num_dof_u0, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices_u0,
                                    orient_indices_u0, elem_restr_u0);
  PetscCall(DMGetLocalVector(dm_p0, &U_loc));
  PetscCall(VecGetLocalSize(U_loc, &num_dof_p0));
  PetscCall(DMRestoreLocalVector(dm_p0, &U_loc));
  CeedElemRestrictionCreate(ceed, num_elem, 1, 1, 1, num_dof_p0, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices_p0, elem_restr_p0);
  PetscCall(PetscFree(restr_indices_p));
  PetscCall(PetscFree(restr_indices_u));
  PetscCall(PetscFree(orient_indices_u));
  PetscCall(PetscFree(restr_indices_u0));
  PetscCall(PetscFree(orient_indices_u0));
  PetscCall(PetscFree(restr_indices_p0));
  PetscFunctionReturn(0);
};
