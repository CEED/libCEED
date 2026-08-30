#include "../include/setup-fe.h"

#include "petscerror.h"
// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

// ---------------------------------------------------------------------------
// Setup FE for H(div) space
// ---------------------------------------------------------------------------
PetscErrorCode SetupFEHdiv(AppCtx app_ctx, ProblemData problem_data, DM dm) {
  PetscSection sec;
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
  // Create section
  PetscCall(PetscSectionCreate(app_ctx->comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, 1));
  PetscCall(PetscSectionSetFieldName(sec, 0, "Velocity"));
  PetscCall(PetscSectionSetFieldComponents(sec, 0, 1));
  PetscCall(PetscSectionSetChart(sec, p_start, p_end));
  // Setup dofs per face
  for (PetscInt f = f_start; f < f_end; f++) {
    PetscCall(DMPlexGetConeSize(dm, f, &dofs_per_face));
    PetscCall(PetscSectionSetFieldDof(sec, f, 0, dofs_per_face));
    PetscCall(PetscSectionSetDof(sec, f, dofs_per_face));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetSection(dm, sec));
  PetscCall(PetscSectionDestroy(&sec));

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
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm, CeedInt P, CeedElemRestriction *elem_restr_u, CeedElemRestriction *elem_restr_p) {
  PetscSection    section;
  PetscInt        p, num_elem, num_dof, *restr_indices_u, *restr_indices_p, elem_offset, num_fields, dim, c_start, c_end;
  Vec             U_loc;
  const PetscInt *ornt;  // this is for orientation of dof

  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &num_fields));
  PetscInt num_comp[num_fields], field_offsets[num_fields + 1];
  field_offsets[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    PetscCall(PetscSectionGetFieldComponents(section, f, &num_comp[f]));
    field_offsets[f + 1] = field_offsets[f] + num_comp[f];
  }
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  PetscCall(PetscMalloc1(num_elem * dim * PetscPowInt(P, dim), &restr_indices_u));
  PetscCall(PetscMalloc1(num_elem, &restr_indices_p));
  bool *orient_indices;  // to flip the dof
  PetscCall(PetscMalloc1(num_elem * dim * PetscPowInt(P, dim), &orient_indices));

  for (p = 0, elem_offset = 0; p < num_elem; p++) {
    restr_indices_p[p] = p;  // each cell has on P0 dof
    PetscInt num_indices, *indices, faces_per_elem, dofs_per_face;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, p, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
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
        orient_indices[elem_offset] = ornt[f] < 0;
        elem_offset++;
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, p, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
  }
  // if (elem_offset != num_elem*dim*PetscPowInt(P, dim))
  //   SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
  //            "ElemRestriction of size (%" PetscInt_FMT ",%" PetscInt_FMT ")
  //             initialized %" PetscInt_FMT "nodes", num_elem,
  //             dim*PetscPowInt(P, dim),elem_offset);
  PetscCall(DMGetLocalVector(dm, &U_loc));
  PetscCall(VecGetLocalSize(U_loc, &num_dof));
  PetscCall(DMRestoreLocalVector(dm, &U_loc));
  // dof per element in Hdiv is dim*P^dim, for linear element P=2
  CeedElemRestrictionCreateOriented(ceed, num_elem, dim * PetscPowInt(P, dim), field_offsets[num_fields], 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                                    restr_indices_u, orient_indices, elem_restr_u);
  CeedElemRestrictionCreate(ceed, num_elem, 1, 1, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices_p, elem_restr_p);
  PetscCall(PetscFree(restr_indices_u));
  PetscCall(PetscFree(orient_indices));
  PetscCall(PetscFree(restr_indices_p));

  PetscFunctionReturn(0);
};