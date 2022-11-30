// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it is included multiple times.

// List each gallery registration function once here.
// This will be expanded inside CeedQFunctionRegisterAll() to call each registration function in the order listed, and also to define weak symbol
// aliases for QFunctions that are not configured.
//
// At the time of this writing, all the gallery functions are defined, but we're adopting the same strategy here as for the backends because future
// gallery QFunctions might depend on external libraries.

MACRO(CeedQFunctionRegister_Identity)
MACRO(CeedQFunctionRegister_Mass1DBuild)
MACRO(CeedQFunctionRegister_Mass2DBuild)
MACRO(CeedQFunctionRegister_Mass3DBuild)
MACRO(CeedQFunctionRegister_MassApply)
MACRO(CeedQFunctionRegister_Vector3MassApply)
MACRO(CeedQFunctionRegister_Poisson1DApply)
MACRO(CeedQFunctionRegister_Poisson1DBuild)
MACRO(CeedQFunctionRegister_Poisson2DApply)
MACRO(CeedQFunctionRegister_Poisson2DBuild)
MACRO(CeedQFunctionRegister_Poisson3DApply)
MACRO(CeedQFunctionRegister_Poisson3DBuild)
MACRO(CeedQFunctionRegister_Vector3Poisson1DApply)
MACRO(CeedQFunctionRegister_Vector3Poisson2DApply)
MACRO(CeedQFunctionRegister_Vector3Poisson3DApply)
MACRO(CeedQFunctionRegister_Scale)
