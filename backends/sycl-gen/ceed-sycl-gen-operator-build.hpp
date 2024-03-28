// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

CEED_INTERN int BlockGridCalculate_Sycl_gen(const CeedInt dim, const CeedInt P_1d, const CeedInt Q_1d, CeedInt *block_sizes);
CEED_INTERN int CeedOperatorBuildKernel_Sycl_gen(CeedOperator op);
