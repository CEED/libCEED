// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

CEED_INTERN int BlockGridCalculate_Hip_gen(CeedInt dim, CeedInt num_elem, CeedInt P_1d, CeedInt Q_1d, CeedInt *block_sizes);
CEED_INTERN int CeedOperatorBuildKernel_Hip_gen(CeedOperator op);
