// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <petsc.h>

extern PetscLogEvent FLUIDS_CeedOperatorApply;
extern PetscLogEvent FLUIDS_CeedOperatorAssemble;
extern PetscLogEvent FLUIDS_CeedOperatorAssembleDiagonal;
extern PetscLogEvent FLUIDS_CeedOperatorAssemblePointBlockDiagonal;
extern PetscLogEvent FLUIDS_SmartRedis_Init;
extern PetscLogEvent FLUIDS_SmartRedis_Meta;
extern PetscLogEvent FLUIDS_SmartRedis_Train;
extern PetscLogEvent FLUIDS_TrainDataCompute;
extern PetscLogEvent FLUIDS_DifferentialFilter;
extern PetscLogEvent FLUIDS_VelocityGradientProjection;
extern PetscLogEvent FLUIDS_SgsModel;
extern PetscLogEvent FLUIDS_SgsModelDDInference;
extern PetscLogEvent FLUIDS_SgsModelDDData;

PetscErrorCode RegisterLogEvents();
