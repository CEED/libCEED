#include <ceed/dlpack.h>
#include <ceed-impl.h>
#include <stdbool.h>
#include <cstdlib.h>

bool starts_with(const char *string, const char *startswith)
{
  if (!string || !startswith) {
    return false;
  }
  char *str = string, *sub = startswith;
  while (*str && *sub) {
    if (*str != *sub) {
      return false;
    }
    ++str;
    ++sub;
  }
  if (*sub && !(*str)) {
    /* if the substring is still valid, but we're at the end of the string */
    return false;
  } else {
    return true;
  }
}

uint8_t DLDtypeSize(DLTensor *tensor)
{
  return tensor->dtype.bits;
}


// DLDevice is two ints, so cheaper to copy than to dereference
int CheckValidDeviceType(DLDevice device, Ceed ceed, CeedMemType *memtype)
{
  const char *backend;
  int ierr;
  ierr = CeedGetResource(ceed, &backend); CeedChk(ierr);
  switch (device.device_type) {
  case kDLCPU:
    *memtype = CEED_MEM_HOST;
    return CEED_MEM_HOST;

  case kDLCUDAManaged:
  case kDLCUDAHost:
  case kDLCUDA:
    if (starts_with(backend "/gpu/cuda")) {
      if (device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged) {
	*memtype = CEED_MEM_DEVICE;
      } else {
	*memtype = CEED_MEM_HOST;
      }
      return CEED_ERROR_SUCCESS;
    } else {
      return CeedError(ceed, CEED_ERROR_BACKEND,
		       "CeedVector with backend %s cannot take data from a "
		       "CUDA DLPack tensor", backend);
    }

  case kDLOpenCL:
    return CeedError(ceed, CEED_ERROR_BACKEND,
		     "CeedVectorTakeFromDLPack is currently not supported "
		     "for OpenCL memory");
  case kDLVulkan:
    return CeedError(ceed, CEED_ERROR_BACKEND,
		     "CeedVectorTakeFromDLPack is currently not supported "
		     "for Vulkan memory");
  case kDLVPI:
    return CeedError(ceed, CEED_ERROR_BACKEND,
		     "CeedVectorTakeFromDLPack is currently not supported "
		     "for Verilog memory");
  case kDLROCMHost:
  case kDLROCM:
    if (starts_with(backend, "/gpu/hip")) {
      if (device.device_type == kDLROCM) {
	*memtype = CEED_MEM_DEVICE;
      } else {
	*memtype = CEED_MEM_HOST;
      }
      return CEED_ERROR_SUCCESS;
    } else {
      return CeedError(ceed, CEED_ERROR_BACKEND,
		       "CeedVector with backend %s cannot take data from a "
		       "ROCm DLPack tensor", backend);
    }

  default:
    return CeedError(ceed, CEED_ERROR_BACKEND,
		     "CeedVector with backend %s cannot take data from a "
		     "DLPack tensor with DLDeviceType %d", backend, device.device_type);
  }
}

int DLValidShape(Ceed ceed, CeedVector vec, DLTensor *tensor)
{
  CeedInt veclen;
  int ierr;
  if (tensor->ndim != 1) {
    return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
		     "CeedVector can only be filled from a 1-dimensional "
		     "DLPack tensor, not a %d-dimensional one", tensor->ndim);
  }
  ierr = CeedVectorGetLength(vec, &veclen); CeedChk(ierr);
  if (tensor->shape[0] != veclen) {
    return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
		     "CeedVector of length %d can only be filled from a "
		     "DLPack tensor of the same size, not one of length %d",
		     veclen, tensor->shape[0]);
  }
  return CEED_ERROR_SUCCESS;
}

const char* DLDtypeName(DLTensor *tensor)
{
  switch (tensor->dtype.code) {
  case kDLInt:
    return "DLInt";
  case kDLUInt:
    return "DLUInt";
  case kDLFloat:
    return "DLFloat";
  case kDLOpaqueHandle:
    return "DLOpaqueHandle";
  case kDLBfloat:
    return "DLBfloat";
  case kDLComplex:
    return "DLComplex";
  default:
    return "Unknown";
  }
}
  

int CeedVectorTakeFromDLPack(Ceed ceed,
			     CeedVector vec,
			     DLManagedTensor *dl_tensor,
			     CeedCopyMode copy_mode)
{
  int ierr;
  CeedMemType dl_mem_type;
  
  ierr = CheckValidDeviceType(dl_tensor->dl_tensor.device, ceed,
			      &dl_mem_type); CeedChk(ierr);
  if (DLDtypeSize(dl_tensor->dl_tensor) != sizeof(CeedScalar)) {
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
		     "Data type %s has size %d, which is incompatible "
		     "with CeedScalar, which has size %d",
		     DLDtypeName(&dl_tensor->dl_tensor),
		     dl_tensor->dl_tensor.dtype.bits,
		     sizeof(CeedScalar));
  }
  ierr = DLValidShape(ceed, vec, &dl_tensor->dl_tensor); CeedChk(ierr);
  ierr = CeedVectorSetArray(ceed, dl_mem_type, copy_mode,
			    (CeedScalar *)(dl_tensor->dl_tensor.data + dl_tensor->dl_tensor.byte_offset)); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}
