// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
// Based on the instructions from https://www.craylabs.org/docs/sr_integration.html and PHASTA implementation

#include "../../include/smartsim.h"

#include "../../navierstokes.h"

PetscErrorCode SmartRedisVerifyPutTensor(void *c_client, const char *name, const size_t name_length) {
  bool does_exist = true;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
  PetscSmartRedisCall(tensor_exists(c_client, name, name_length, &does_exist));
  PetscCheck(does_exist, PETSC_COMM_SELF, -1, "Tensor of name '%s' was not written to the database successfully", name);
  PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SmartSimTrainingSetup(User user) {
  SmartSimData smartsim = user->smartsim;
  PetscMPIInt  rank;
  PetscReal    checkrun[2] = {1};
  size_t       dim_2[1]    = {2};

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(user->comm, &rank));

  if (rank % smartsim->collocated_database_num_ranks == 0) {
    // -- Send array that communicates when ML is done training
    PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
    PetscSmartRedisCall(put_tensor(smartsim->client, "check-run", 9, checkrun, dim_2, 1, SRTensorTypeDouble, SRMemLayoutContiguous));
    PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "check-run", 9));
    PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Meta, 0, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SmartSimSetup(User user) {
  PetscMPIInt rank;
  PetscInt    num_orchestrator_nodes = 1;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user->smartsim));
  SmartSimData smartsim = user->smartsim;

  smartsim->collocated_database_num_ranks = 1;
  PetscOptionsBegin(user->comm, NULL, "Options for SmartSim integration", NULL);
  PetscCall(PetscOptionsInt("-smartsim_collocated_database_num_ranks", "Number of ranks per collocated database instance", NULL,
                            smartsim->collocated_database_num_ranks, &smartsim->collocated_database_num_ranks, NULL));
  PetscOptionsEnd();

  // Create prefix to be put on tensor names
  PetscCallMPI(MPI_Comm_rank(user->comm, &rank));
  PetscCall(PetscSNPrintf(smartsim->rank_id_name, sizeof(smartsim->rank_id_name), "y.%d", rank));

  PetscCall(PetscLogEventBegin(FLUIDS_SmartRedis_Init, 0, 0, 0, 0));
  PetscSmartRedisCall(SmartRedisCClient(num_orchestrator_nodes != 1, smartsim->rank_id_name, strlen(smartsim->rank_id_name), &smartsim->client));
  PetscCall(PetscLogEventEnd(FLUIDS_SmartRedis_Init, 0, 0, 0, 0));

  PetscCall(SmartSimTrainingSetup(user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SmartSimDataDestroy(SmartSimData smartsim) {
  PetscFunctionBeginUser;
  if (!smartsim) PetscFunctionReturn(PETSC_SUCCESS);

  PetscSmartRedisCall(DeleteCClient(&smartsim->client));
  PetscCall(PetscFree(smartsim));
  PetscFunctionReturn(PETSC_SUCCESS);
}
