#include <ceed/dlpack.h>
#include <ceed/backend.h>
#include <ceed-impl.h>
#include <stdbool.h>
#include <stdlib.h>

bool starts_with(const char *string, const char *startswith)
{
  if (!string || !startswith) {
    return false;
  }
  const char *str = string, *sub = startswith;
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
  return tensor->dtype.bits / 8;
}

DLDeviceType GetDLDeviceType(Ceed ceed, CeedMemType mem_type)
{
  const char *backend;
  int ierr;
  ierr = CeedGetResource(ceed, &backend); CeedChk(ierr);
  if (mem_type == CEED_MEM_HOST) {
    return kDLCPU;
  }
  else {
    if (starts_with(backend, "/gpu/cuda")) {
      return kDLCUDA;
    } else if (starts_with(backend, "/gpu/hip") || starts_with(backend, "/gpu/occa")) {
      return kDLROCM;
    } else {
      return kDLExtDev; /* unknown */
    }
  }
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
    if (starts_with(backend, "/gpu/cuda") || starts_with(backend, "/gpu/occa")) {
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
    if (starts_with(backend, "/gpu/occa")) {
      return CEED_ERROR_SUCCESS;
    }
    return CeedError(ceed, CEED_ERROR_BACKEND,
		     "CeedVectorTakeFromDLPack is currently not supported "
		     "for OpenCL memory");
  case kDLVulkan:
    if (starts_with(backend, "/gpu/occa")) {
      return CEED_ERROR_SUCCESS;
    }
    return CeedError(ceed, CEED_ERROR_BACKEND,
		     "CeedVectorTakeFromDLPack is currently not supported "
		     "for Vulkan memory");
  case kDLVPI:
    return CeedError(ceed, CEED_ERROR_BACKEND,
		     "CeedVectorTakeFromDLPack is currently not supported "
		     "for Verilog memory");
  case kDLROCMHost:
  case kDLROCM:
    if (starts_with(backend, "/gpu/hip") || starts_with(backend, "/gpu/occa")) {
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
  if (tensor->dtype.lanes != 1) {
    return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
		     "CeedVector can only be filled from a DLPack tensor with "
		     "scalar elements (1 lane), not one with %d lanes", tensor->dtype.lanes);
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
  if (DLDtypeSize(&dl_tensor->dl_tensor) != sizeof(CeedScalar)) {
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
		     "Data type %s has size %d, which is incompatible "
		     "with CeedScalar, which has size %d",
		     DLDtypeName(&dl_tensor->dl_tensor),
		     dl_tensor->dl_tensor.dtype.bits / 8,
		     sizeof(CeedScalar));
  }
  ierr = DLValidShape(ceed, vec, &dl_tensor->dl_tensor); CeedChk(ierr);
  ierr = CeedVectorSetArray(vec, dl_mem_type, copy_mode,
			    (CeedScalar *)(dl_tensor->dl_tensor.data + dl_tensor->dl_tensor.byte_offset)); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}


void CeedDLPackDeleter(struct DLManagedTensor *self)
{
  CeedVector vec = (CeedVector)self->manager_ctx;
  CeedScalar *array = (CeedScalar*)(self->dl_tensor.data + self->dl_tensor.byte_offset);
  CeedVectorRestoreArray(vec, &array);
  CeedFree(&(self->dl_tensor.shape));
  CeedVectorDestroy(&vec);
}

int CeedVectorToDLPack(Ceed ceed,
		       CeedVector vec,
		       CeedMemType dl_mem_type,
		       DLManagedTensor *tensor)
{
  int ierr;
  CeedInt veclen;
  CeedScalar *array;
  const char *backend;
  tensor->manager_ctx = (void*)vec;
  tensor->deleter = CeedDLPackDeleter;
  ierr = CeedVectorGetArray(vec, dl_mem_type, &array); CeedChk(ierr);
  tensor->dl_tensor.data = (void *)array;
  tensor->dl_tensor.byte_offset = 0;
  tensor->dl_tensor.dtype.code = sizeof(CeedScalar) <= 8 ? kDLFloat : kDLComplex;
  tensor->dl_tensor.dtype.bits = 8 * sizeof(CeedScalar);
  tensor->dl_tensor.dtype.lanes = 1;
  
  ierr = CeedVectorGetLength(vec, &veclen); CeedChk(ierr);
  tensor->dl_tensor.ndim = 1;
  ierr = CeedMalloc(1, &(tensor->dl_tensor.shape)); CeedChk(ierr);
  tensor->dl_tensor.shape[0] = veclen;
  tensor->dl_tensor.strides = NULL;

  DLDeviceType devtype = GetDLDeviceType(ceed, dl_mem_type);
  if (devtype == kDLExtDev) {
    ierr = CeedGetResource(ceed, &backend); CeedChk(ierr);
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
		     "Backend %s does not currently have DLPack support",
		     backend);
  }
  tensor->dl_tensor.device.device_type = devtype;
  return CEED_ERROR_SUCCESS;
}
