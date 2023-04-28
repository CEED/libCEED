// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_gen_operator_build_h
#define _ceed_sycl_gen_operator_build_h

CEED_INTERN int BlockGridCalculate_Sycl_gen(const CeedInt dim, const CeedInt num_elem, const CeedInt P_1d, const CeedInt Q_1d, CeedInt *block_sizes);
CEED_INTERN int CeedSyclGenOperatorBuild(CeedOperator op);

#endif  // _ceed_sycl_gen_operator_build_h
