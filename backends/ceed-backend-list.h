// This header does not have guards because it is included multiple times.

// List each backend registration function once here. This will be expanded
// inside CeedRegisterAll() to call each registration function in the order
// listed, and also to define weak symbol aliases for backends that are not
// configured.

MACRO(CeedRegister_Avx_Blocked)
MACRO(CeedRegister_Avx_Serial)
MACRO(CeedRegister_Cuda)
MACRO(CeedRegister_Cuda_Gen)
MACRO(CeedRegister_Cuda_Shared)
MACRO(CeedRegister_Hip)
MACRO(CeedRegister_Hip_Gen)
MACRO(CeedRegister_Hip_Shared)
MACRO(CeedRegister_Magma)
MACRO(CeedRegister_Magma_Det)
MACRO(CeedRegister_Memcheck_Blocked)
MACRO(CeedRegister_Memcheck_Serial)
MACRO(CeedRegister_Occa)
MACRO(CeedRegister_Opt_Blocked)
MACRO(CeedRegister_Opt_Serial)
MACRO(CeedRegister_Ref)
MACRO(CeedRegister_Ref_Blocked)
MACRO(CeedRegister_Tmpl)
MACRO(CeedRegister_Tmpl_Sub)
MACRO(CeedRegister_Xsmm_Blocked)
MACRO(CeedRegister_Xsmm_Serial)
