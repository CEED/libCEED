// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_hip_gen_operator_build_h
#define _ceed_hip_gen_operator_build_h

CEED_INTERN int BlockGridCalculate_Hip_gen(CeedInt dim, CeedInt num_elem, CeedInt P_1d, CeedInt Q_1d, CeedInt *block_sizes);
CEED_INTERN int CeedHipGenOperatorBuild(CeedOperator op);

#endif  // _ceed_hip_gen_operator_build_h
