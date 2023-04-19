// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
// Based on the instructions from https://www.craylabs.org/docs/sr_integration.html and PHASTA implementation

#include "../include/smartsim.h"

#include "../navierstokes.h"

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

  {  // -- Get minimum per-rank global vec size
    PetscInt GlobalVecSize;
    PetscCall(DMGetGlobalVectorInfo(user->dm, &GlobalVecSize, NULL, NULL));
    PetscCallMPI(MPI_Allreduce(&GlobalVecSize, &smartsim->num_tensor_nodes, 1, MPIU_INT, MPI_MIN, user->comm));
    smartsim->num_nodes_to_remove = GlobalVecSize - smartsim->num_tensor_nodes;
  }

  // Determine the size of the training data arrays... somehow
  if (rank % smartsim->collocated_database_num_ranks == 0) {
    size_t   array_dims[2] = {smartsim->num_tensor_nodes, 6}, array_info_dim = 6;
    PetscInt array_info[6] = {0}, num_features = 6;

    array_info[0] = array_dims[0];
    array_info[1] = array_dims[1];
    array_info[2] = num_features;
    array_info[3] = num_ranks;
    array_info[4] = smartsim->collocated_database_num_ranks;
    array_info[5] = rank;

    SmartRedisCall(put_tensor(smartsim->client, "array_info", 10, array_info, &array_info_dim, 1, SRTensorTypeInt32, SRMemLayoutContiguous));
    PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "array_info", 10));
  }
  PetscFunctionReturn(0);
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

  PetscFunctionReturn(0);
}

PetscErrorCode TSMonitor_SmartSimTraining(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx) {
  User user = (User)ctx;
  Vec  FilteredFields;
  PetscFunctionBeginUser;

  PetscCall(DMGetGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
  PetscCall(DMRestoreGlobalVector(user->diff_filter->dm_filter, &FilteredFields));
  PetscFunctionReturn(0);
}
