// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  SmartRedisCall(tensor_exists(c_client, name, name_length, &does_exist));
  PetscCheck(does_exist, PETSC_COMM_SELF, -1, "Tensor of name '%s' was not written to the database successfully", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SmartSimTrainingSetup(User user) {
  SmartSimData smartsim = user->smartsim;
  PetscMPIInt  rank;
  PetscReal    checkrun[2] = {1};
  size_t       dim_2[1]    = {2};
  PetscInt     num_ranks;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(user->comm, &rank));
  PetscCallMPI(MPI_Comm_size(user->comm, &num_ranks));

  if (rank % smartsim->collocated_database_num_ranks == 0) {
    // -- Send array that communicates when ML is done training
    SmartRedisCall(put_tensor(smartsim->client, "check-run", 9, checkrun, dim_2, 1, SRTensorTypeDouble, SRMemLayoutContiguous));
    PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "check-run", 9));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SmartSimSetup(User user) {
  PetscMPIInt rank;
  size_t      rank_id_name_len;
  PetscInt    num_orchestrator_nodes = 1;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user->smartsim));
  SmartSimData smartsim = user->smartsim;

  smartsim->collocated_database_num_ranks = 1;
  PetscOptionsBegin(user->comm, NULL, "Options for SmartSim integration", NULL);
  PetscCall(PetscOptionsInt("-smartsim_collocated_database_num_ranks", "Number of ranks per collocated database instance", NULL,
                            smartsim->collocated_database_num_ranks, &smartsim->collocated_database_num_ranks, NULL));
  PetscOptionsEnd();

  PetscCall(PetscStrlen(smartsim->rank_id_name, &rank_id_name_len));
  // Create prefix to be put on tensor names
  PetscCallMPI(MPI_Comm_rank(user->comm, &rank));
  PetscCall(PetscSNPrintf(smartsim->rank_id_name, sizeof smartsim->rank_id_name, "y.%d", rank));

  SmartRedisCall(SmartRedisCClient(num_orchestrator_nodes != 1, smartsim->rank_id_name, rank_id_name_len, &smartsim->client));

  PetscCall(SmartSimTrainingSetup(user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SmartSimDataDestroy(SmartSimData smartsim) {
  PetscFunctionBeginUser;
  if (!smartsim) PetscFunctionReturn(PETSC_SUCCESS);

  SmartRedisCall(DeleteCClient(&smartsim->client));
  PetscCall(PetscFree(smartsim));
  PetscFunctionReturn(PETSC_SUCCESS);
}
