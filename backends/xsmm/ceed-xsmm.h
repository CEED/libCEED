// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_xsmm_h
#define _ceed_xsmm_h

#include <ceed.h>
#include <ceed/backend.h>

CEED_INTERN int CeedTensorContractCreate_Xsmm(CeedBasis basis, CeedTensorContract contract);

#endif  // _ceed_xsmm_h
