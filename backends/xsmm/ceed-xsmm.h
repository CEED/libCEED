// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_XSMM_H
#define CEED_XSMM_H

#include <ceed.h>
#include <ceed/backend.h>

CEED_INTERN int CeedTensorContractCreate_Xsmm(CeedBasis basis, CeedTensorContract contract);

#endif  // CEED_XSMM_H
