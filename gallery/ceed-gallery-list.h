// This header does not have guards because it is included multiple times.

// List each gallery registration function once here. This will be expanded
// inside CeedQFunctionRegisterAll() to call each registration function in the
// order listed, and also to define weak symbol aliases for backends that are
// not configured.
//
// At the time of this writing, all the gallery functions are defined, but we're
// adopting the same strategy here as for the backends because future gallery
// functions might depend on external libraries.

MACRO(CeedQFunctionRegister_Identity)
MACRO(CeedQFunctionRegister_Mass1DBuild)
MACRO(CeedQFunctionRegister_Mass2DBuild)
MACRO(CeedQFunctionRegister_Mass3DBuild)
MACRO(CeedQFunctionRegister_MassApply)
MACRO(CeedQFunctionRegister_Poisson1DApply)
MACRO(CeedQFunctionRegister_Poisson1DBuild)
MACRO(CeedQFunctionRegister_Poisson2DApply)
MACRO(CeedQFunctionRegister_Poisson2DBuild)
MACRO(CeedQFunctionRegister_Poisson3DApply)
MACRO(CeedQFunctionRegister_Poisson3DBuild)
MACRO(CeedQFunctionRegister_Scale)
