// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// This header does not have guards because it is included multiple times.

// List each gallery registration function once here.
// This will be expanded inside @ref CeedQFunctionRegisterAll() to call each registration function in the order listed, and also to define weak symbol aliases for @ref CeedQFunction that are not configured.
//
// At the time of this writing, all the gallery functions are defined, but we're adopting the same strategy here as for the backends because future gallery @ref CeedQFunction might depend on external libraries.

CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Identity)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Mass1DBuild)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Mass2DBuild)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Mass3DBuild)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_MassApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Vector3MassApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Poisson1DApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Poisson1DBuild)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Poisson2DApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Poisson2DBuild)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Poisson3DApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Poisson3DBuild)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Vector3Poisson1DApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Vector3Poisson2DApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Vector3Poisson3DApply)
CEED_GALLERY_QFUNCTION(CeedQFunctionRegister_Scale)
