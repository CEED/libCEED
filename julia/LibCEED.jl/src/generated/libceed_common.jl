# Automatically generated using Clang.jl
#! format: off

const FILE = Cvoid

# Skipping MacroDefinition: CEED_QFUNCTION ( name ) static const char name ## _loc [ ] = __FILE__ ":" # name ; static int name

# Skipping MacroDefinition: CeedError ( ceed , ecode , ... ) ( CeedErrorImpl ( ( ceed ) , __FILE__ , __LINE__ , __func__ , ( ecode ) , __VA_ARGS__ ) ? : ( ecode ) )

const CEED_VERSION_MAJOR = 0
const CEED_VERSION_MINOR = 8
const CEED_VERSION_PATCH = 0
const CEED_VERSION_RELEASE = false

# Skipping MacroDefinition: CEED_VERSION_GE ( major , minor , patch ) ( ! CEED_VERSION_RELEASE || ( CEED_VERSION_MAJOR > major || ( CEED_VERSION_MAJOR == major && ( CEED_VERSION_MINOR > minor || ( CEED_VERSION_MINOR == minor && CEED_VERSION_PATCH >= patch ) ) ) ) )

const CeedInt = Int32
const CeedScalar = Cdouble
const Ceed_private = Cvoid
const Ceed = Ptr{Ceed_private}
const CeedRequest_private = Cvoid
const CeedRequest = Ptr{CeedRequest_private}
const CeedVector_private = Cvoid
const CeedVector = Ptr{CeedVector_private}
const CeedElemRestriction_private = Cvoid
const CeedElemRestriction = Ptr{CeedElemRestriction_private}
const CeedBasis_private = Cvoid
const CeedBasis = Ptr{CeedBasis_private}
const CeedQFunction_private = Cvoid
const CeedQFunction = Ptr{CeedQFunction_private}
const CeedQFunctionContext_private = Cvoid
const CeedQFunctionContext = Ptr{CeedQFunctionContext_private}
const CeedOperator_private = Cvoid
const CeedOperator = Ptr{CeedOperator_private}
const CeedErrorHandler = Ptr{Cvoid}

@cenum CeedErrorType::Int32 begin
    CEED_ERROR_SUCCESS = 0
    CEED_ERROR_MINOR = 1
    CEED_ERROR_DIMENSION = 2
    CEED_ERROR_INCOMPLETE = 3
    CEED_ERROR_INCOMPATIBLE = 4
    CEED_ERROR_ACCESS = 5
    CEED_ERROR_MAJOR = -1
    CEED_ERROR_BACKEND = -2
    CEED_ERROR_UNSUPPORTED = -3
end

@cenum CeedMemType::UInt32 begin
    CEED_MEM_HOST = 0
    CEED_MEM_DEVICE = 1
end

@cenum CeedCopyMode::UInt32 begin
    CEED_COPY_VALUES = 0
    CEED_USE_POINTER = 1
    CEED_OWN_POINTER = 2
end

@cenum CeedNormType::UInt32 begin
    CEED_NORM_1 = 0
    CEED_NORM_2 = 1
    CEED_NORM_MAX = 2
end

@cenum CeedTransposeMode::UInt32 begin
    CEED_NOTRANSPOSE = 0
    CEED_TRANSPOSE = 1
end

@cenum CeedEvalMode::UInt32 begin
    CEED_EVAL_NONE = 0
    CEED_EVAL_INTERP = 1
    CEED_EVAL_GRAD = 2
    CEED_EVAL_DIV = 4
    CEED_EVAL_CURL = 8
    CEED_EVAL_WEIGHT = 16
end

@cenum CeedQuadMode::UInt32 begin
    CEED_GAUSS = 0
    CEED_GAUSS_LOBATTO = 1
end

@cenum CeedElemTopology::UInt32 begin
    CEED_LINE = 65536
    CEED_TRIANGLE = 131073
    CEED_QUAD = 131074
    CEED_TET = 196611
    CEED_PYRAMID = 196612
    CEED_PRISM = 196613
    CEED_HEX = 196614
end


const CeedQFunctionUser = Ptr{Cvoid}

# Skipping MacroDefinition: CEED_INTERN CEED_EXTERN __attribute__ ( ( visibility ( "hidden" ) ) )
# Skipping MacroDefinition: CEED_UNUSED __attribute__ ( ( unused ) )

const CEED_MAX_RESOURCE_LEN = 1024
const CEED_MAX_BACKEND_PRIORITY = typemax(Cuint)
const CEED_ALIGN = 64
const CEED_COMPOSITE_MAX = 16
const CEED_EPSILON = 1.0e-16
const CEED_DEBUG_COLOR = 0

# Skipping MacroDefinition: CeedDebug1 ( ceed , format , ... ) CeedDebugImpl ( ceed , format , ## __VA_ARGS__ )
# Skipping MacroDefinition: CeedDebug256 ( ceed , color , ... ) CeedDebugImpl256 ( ceed , color , ## __VA_ARGS__ )
# Skipping MacroDefinition: CeedDebug ( ... ) CeedDebug256 ( ceed , ( unsigned char ) CEED_DEBUG_COLOR , ## __VA_ARGS__ )
# Skipping MacroDefinition: CeedChk ( ierr ) do { int ierr_ = ierr ; if ( ierr_ ) return ierr_ ; } while ( 0 )
# Skipping MacroDefinition: CeedChkBackend ( ierr ) do { int ierr_ = ierr ; if ( ierr_ ) { if ( ierr_ > CEED_ERROR_SUCCESS ) return CEED_ERROR_BACKEND ; else return ierr_ ; } } while ( 0 )
# Skipping MacroDefinition: CeedMalloc ( n , p ) CeedMallocArray ( ( n ) , sizeof ( * * ( p ) ) , p )
# Skipping MacroDefinition: CeedCalloc ( n , p ) CeedCallocArray ( ( n ) , sizeof ( * * ( p ) ) , p )
# Skipping MacroDefinition: CeedRealloc ( n , p ) CeedReallocArray ( ( n ) , sizeof ( * * ( p ) ) , p )

const CeedTensorContract_private = Cvoid
const CeedTensorContract = Ptr{CeedTensorContract_private}
const CeedQFunctionField_private = Cvoid
const CeedQFunctionField = Ptr{CeedQFunctionField_private}
const CeedOperatorField_private = Cvoid
const CeedOperatorField = Ptr{CeedOperatorField_private}
