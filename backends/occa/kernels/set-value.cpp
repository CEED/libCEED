#include "./kernel-defines.hpp"

// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - BLOCK_SIZE : CeedInt

const char *occa_set_value_source = STRINGIFY_SOURCE(

  @kernel 
  void setValue(CeedScalar* ptr,const CeedScalar value,const CeedInt count) {
    @tile(BLOCK_SIZE,@outer,@inner)
    for(CeedInt i=0; i < count; ++i) {
      ptr[i] = value;
    }
  }
);
