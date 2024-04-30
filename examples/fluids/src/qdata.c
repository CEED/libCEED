// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../navierstokes.h"

#include <petscsection.h>
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/setupgeo2d.h"

/**
 * @brief Get number of components of quadrature data for domain
 *
 * @param[in]  dm          DM where quadrature data would be used
 * @param[out] q_data_size Number of components of quadrature data
 */
PetscErrorCode QDataGetNumComponents(DM dm, CeedInt *q_data_size) {
  PetscInt num_comp_x, dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  {  // Get number of coordinate components
    DM           dm_coord;
    PetscSection section_coord;
    PetscInt     field = 0;  // Default field has the coordinates
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMGetLocalSection(dm_coord, &section_coord));
    PetscCall(PetscSectionGetFieldComponents(section_coord, field, &num_comp_x));
  }
  switch (dim) {
    case 2:
      switch (num_comp_x) {
        case 2:
          *q_data_size = 5;
          break;
        case 3:
          *q_data_size = 7;
          break;
        default:
          SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,
                  "QData not valid for DM of dimension %" PetscInt_FMT " and coordinates with dimension %" PetscInt_FMT, dim, num_comp_x);
          break;
      }
      break;
    case 3:
      *q_data_size = 10;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,
              "QData not valid for DM of dimension %" PetscInt_FMT " and coordinates with dimension %" PetscInt_FMT, dim, num_comp_x);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Create quadrature data for domain
 *
 * @param[in]  ceed          Ceed object quadrature data will be used with
 * @param[in]  dm            DM where quadrature data would be used
 * @param[in]  domain_label  DMLabel that quadrature data would be used one
 * @param[in]  label_value   Value of label
 * @param[in]  elem_restr_x  CeedElemRestriction of the coordinates (must match `domain_label` and `label_value` selections)
 * @param[in]  basis_x       CeedBasis of the coordinates
 * @param[in]  x_coord       CeedVector of the coordinates
 * @param[out] elem_restr_qd CeedElemRestriction of the quadrature data
 * @param[out] q_data        CeedVector of the quadrature data
 * @param[out] q_data_size   number of components of quadrature data
 */
PetscErrorCode QDataGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                        CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size) {
  CeedQFunction qf_setup;
  CeedOperator  op_setup;
  CeedInt       num_comp_x;
  PetscInt      dim, height = 0;

  PetscFunctionBeginUser;
  PetscCall(QDataGetNumComponents(dm, q_data_size));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x));
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
    case 2:
      switch (num_comp_x) {
        case 2:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Setup2d, Setup2d_loc, &qf_setup));
          break;
        case 3:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Setup2D_3Dcoords, Setup2D_3Dcoords_loc, &qf_setup));
          break;
      }
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Setup, Setup_loc, &qf_setup));
      break;
  }

  // -- Create QFunction for quadrature data
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_setup, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "dx", num_comp_x * (dim - height), CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "surface qdata", *q_data_size, CEED_EVAL_NONE));

  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, *q_data_size, elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_qd, q_data, NULL));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "surface qdata", *elem_restr_qd, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorApply(op_setup, x_coord, *q_data, CEED_REQUEST_IMMEDIATE));

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Get number of components of quadrature data for boundary of domain
 *
 * @param[in]  dm          DM where quadrature data would be used
 * @param[out] q_data_size Number of components of quadrature data
 */
PetscErrorCode QDataBoundaryGetNumComponents(DM dm, CeedInt *q_data_size) {
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
    case 2:
      *q_data_size = 3;
      break;
    case 3:
      *q_data_size = 10;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "QDataBoundary not valid for DM of dimension %" PetscInt_FMT, dim);
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Create quadrature data for boundary of domain
 *
 * @param[in]  ceed          Ceed object quadrature data will be used with
 * @param[in]  dm            DM where quadrature data would be used
 * @param[in]  domain_label  DMLabel that quadrature data would be used one
 * @param[in]  label_value   Value of label
 * @param[in]  elem_restr_x  CeedElemRestriction of the coordinates (must match `domain_label` and `label_value` selections)
 * @param[in]  basis_x       CeedBasis of the coordinates
 * @param[in]  x_coord       CeedVector of the coordinates
 * @param[out] elem_restr_qd CeedElemRestriction of the quadrature data
 * @param[out] q_data        CeedVector of the quadrature data
 * @param[out] q_data_size   number of components of quadrature data
 */
PetscErrorCode QDataBoundaryGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                                CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size) {
  CeedQFunction qf_setup_sur;
  CeedOperator  op_setup_sur;
  CeedInt       num_comp_x;
  PetscInt      dim, height = 1;

  PetscFunctionBeginUser;
  PetscCall(QDataBoundaryGetNumComponents(dm, q_data_size));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x));
  PetscCall(DMGetDimension(dm, &dim));
  switch (dim) {
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupBoundary2d, SetupBoundary2d_loc, &qf_setup_sur));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupBoundary, SetupBoundary_loc, &qf_setup_sur));
      break;
  }

  // -- Create QFunction for quadrature data
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_setup_sur, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup_sur, "dx", num_comp_x * (dim - height), CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup_sur, "weight", 1, CEED_EVAL_WEIGHT));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup_sur, "surface qdata", *q_data_size, CEED_EVAL_NONE));

  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, dm, domain_label, label_value, height, *q_data_size, elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_qd, q_data, NULL));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup_sur, NULL, NULL, &op_setup_sur));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_setup_sur, "surface qdata", *elem_restr_qd, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorApply(op_setup_sur, x_coord, *q_data, CEED_REQUEST_IMMEDIATE));

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup_sur));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup_sur));
  PetscFunctionReturn(PETSC_SUCCESS);
}
