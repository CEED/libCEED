// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <log_events.h>
#include <petsc.h>

static PetscClassId libCEED_classid, misc_classid;

PetscLogEvent FLUIDS_CeedOperatorApply;
PetscLogEvent FLUIDS_CeedOperatorAssemble;
PetscLogEvent FLUIDS_CeedOperatorAssembleDiagonal;
PetscLogEvent FLUIDS_CeedOperatorAssemblePointBlockDiagonal;
PetscLogEvent FLUIDS_SmartRedis_Init;
PetscLogEvent FLUIDS_SmartRedis_Meta;
PetscLogEvent FLUIDS_SmartRedis_Train;
PetscLogEvent FLUIDS_TrainDataCompute;
PetscLogEvent FLUIDS_DifferentialFilter;
PetscLogEvent FLUIDS_VelocityGradientProjection;

PetscErrorCode RegisterLogEvents() {
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("libCEED", &libCEED_classid));
  PetscCall(PetscLogEventRegister("CeedOpApply", libCEED_classid, &FLUIDS_CeedOperatorApply));
  PetscCall(PetscLogEventRegister("CeedOpAsm", libCEED_classid, &FLUIDS_CeedOperatorAssemble));
  PetscCall(PetscLogEventRegister("CeedOpAsmD", libCEED_classid, &FLUIDS_CeedOperatorAssembleDiagonal));
  PetscCall(PetscLogEventRegister("CeedOpAsmPBD", libCEED_classid, &FLUIDS_CeedOperatorAssemblePointBlockDiagonal));

  PetscCall(PetscClassIdRegister("Miscellaneous", &misc_classid));
  PetscCall(PetscLogEventRegister("DiffFilter", misc_classid, &FLUIDS_DifferentialFilter));
  PetscCall(PetscLogEventRegister("VeloGradProj", misc_classid, &FLUIDS_VelocityGradientProjection));
  PetscFunctionReturn(PETSC_SUCCESS);
}
