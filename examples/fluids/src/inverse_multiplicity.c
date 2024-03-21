// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../qfunctions/inverse_multiplicity.h"
#include "../navierstokes.h"

/**
 * @brief Get the inverse of the node multiplicity
 *
 * Local multiplicity concerns only the number of elements associated with a node in the current rank's partition.
 * Global multiplicity is the multiplicity for the global mesh, including periodicity.
 *
 * @param[in]  ceed                        `Ceed` object for the output `CeedVector` and `CeedElemRestriction`
 * @param[in]  dm                          `DM` for the grid
 * @param[in]  domain_label                `DMLabel` for `DMPlex` domain
 * @param[in]  label_value                 Stratum value
 * @param[in]  height                      Height of `DMPlex` topology
 * @param[in]  dm_field                    Index of `DMPlex` field
 * @param[in]  get_global_multiplicity     Whether the multiplicity should be global or local
 * @param[out] elem_restr_inv_multiplicity `CeedElemRestriction` needed to use the multiplicity vector
 * @param[out] inv_multiplicity            `CeedVector` containing the inverse of the multiplicity
 */
PetscErrorCode GetInverseMultiplicity(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                      PetscBool get_global_multiplicity, CeedElemRestriction *elem_restr_inv_multiplicity,
                                      CeedVector *inv_multiplicity) {
  Vec                 Multiplicity, Multiplicity_loc;
  PetscMemType        m_mem_type;
  CeedVector          multiplicity;
  CeedQFunction       qf_multiplicity;
  CeedOperator        op_multiplicity;
  CeedInt             num_comp;
  CeedElemRestriction elem_restr;

  PetscFunctionBeginUser;

  PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, domain_label, label_value, height, 1, elem_restr_inv_multiplicity));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_inv_multiplicity, inv_multiplicity, NULL));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, dm_field, &elem_restr));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr, &num_comp));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr, &multiplicity, NULL));

  if (get_global_multiplicity) {
    // In order to get global multiplicity, we need to run through DMLocalToGlobal -> DMGlobalToLocal.
    PetscCall(DMGetLocalVector(dm, &Multiplicity_loc));
    PetscCall(DMGetGlobalVector(dm, &Multiplicity));

    PetscCall(VecPetscToCeed(Multiplicity_loc, &m_mem_type, multiplicity));
    PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(elem_restr, multiplicity));
    PetscCall(VecCeedToPetsc(multiplicity, m_mem_type, Multiplicity_loc));
    PetscCall(VecZeroEntries(Multiplicity));
    PetscCall(DMLocalToGlobal(dm, Multiplicity_loc, ADD_VALUES, Multiplicity));
    PetscCall(DMGlobalToLocal(dm, Multiplicity, INSERT_VALUES, Multiplicity_loc));
    PetscCall(VecPetscToCeed(Multiplicity_loc, &m_mem_type, multiplicity));
  } else {
    PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(elem_restr, multiplicity));
  }

  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, InverseMultiplicity, InverseMultiplicity_loc, &qf_multiplicity));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_multiplicity, "multiplicity", num_comp, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_multiplicity, "inverse multiplicity", 1, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_multiplicity, NULL, NULL, &op_multiplicity));
  PetscCallCeed(ceed, CeedOperatorSetName(op_multiplicity, "InverseMultiplicity"));
  PetscCallCeed(ceed, CeedOperatorSetField(op_multiplicity, "multiplicity", elem_restr, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_multiplicity, "inverse multiplicity", *elem_restr_inv_multiplicity, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorApply(op_multiplicity, multiplicity, *inv_multiplicity, CEED_REQUEST_IMMEDIATE));

  if (get_global_multiplicity) {
    PetscCall(VecCeedToPetsc(multiplicity, m_mem_type, Multiplicity_loc));
    PetscCall(DMRestoreLocalVector(dm, &Multiplicity_loc));
    PetscCall(DMRestoreGlobalVector(dm, &Multiplicity));
  }
  PetscCallCeed(ceed, CeedVectorDestroy(&multiplicity));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_multiplicity));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_multiplicity));
  PetscFunctionReturn(PETSC_SUCCESS);
}
