#include <ceed/backend.h>
#include <ceed-impl.h>
#include <stdbool.h>
#include <stdlib.h>
//#include <Python.h>
#define CEED_DLPACK_MODULE
#include <ceed/dlpack.h>

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
      return kDLGPU;
    } else if (starts_with(backend, "/gpu/hip") || starts_with(backend, "/gpu/occa")) {
      return kDLROCM;
    } else {
      return kDLExtDev; /* unknown */
    }
  }
}

// DLDevice is two ints, so cheaper to copy than to dereference
int CheckValidDeviceType(DLContext device, Ceed ceed, CeedMemType *memtype)
{
  const char *backend;
  int ierr;
  ierr = CeedGetResource(ceed, &backend); CeedChk(ierr);
  switch (device.device_type) {
  case kDLCPUPinned:
  case kDLCPU:
    *memtype = CEED_MEM_HOST;
    return CEED_MEM_HOST;

  case kDLGPU:
    if (starts_with(backend, "/gpu/cuda") || starts_with(backend, "/gpu/occa")) {
      *memtype = CEED_MEM_DEVICE;
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
  case kDLROCM:
    if (starts_with(backend, "/gpu/hip") || starts_with(backend, "/gpu/occa")) {
      *memtype = CEED_MEM_HOST;
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
  case kDLBfloat:
    return "DLBfloat";
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
  
  ierr = CheckValidDeviceType(dl_tensor->dl_tensor.ctx, ceed,
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
  CeedVectorSetArray(vec, (self->dl_tensor.ctx.device_type == kDLCPU) ?
		     CEED_MEM_HOST : CEED_MEM_DEVICE, CEED_USE_POINTER, array);
  CeedFree(&(self->dl_tensor.shape));
  CeedFree(&(self->dl_tensor.data));
  //CeedVectorDestroy(&vec);
  //;
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
  ierr = CeedVectorGetLength(vec, &veclen); CeedChk(ierr);
  ierr = CeedVectorTakeArray(vec, dl_mem_type, &array); CeedChk(ierr);
  tensor->dl_tensor.data = (void *)array;
  tensor->dl_tensor.byte_offset = 0;
  tensor->dl_tensor.dtype.code = kDLFloat;
  tensor->dl_tensor.dtype.bits = 8 * sizeof(CeedScalar);
  tensor->dl_tensor.dtype.lanes = 1;
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
  tensor->dl_tensor.ctx.device_type = devtype;
  tensor->dl_tensor.ctx.device_id = 0;
  return CEED_ERROR_SUCCESS;
}

int CeedPrintDLManagedTensor(DLManagedTensor *dl_tensor)
{
  char shape[23 * dl_tensor->dl_tensor.ndim + 3]; /* space for tuple of comma-separated values and null terminator, assuming the dimensions might be enormous and take up to 20 digits (which fits a whole int64_t regardless of value) */
  char strides[23 * dl_tensor->dl_tensor.ndim + 3];
  int stride_offset, shape_offset;
  stride_offset = shape_offset = 0;
  shape_offset = snprintf(shape, 2, "[");
  if (dl_tensor->dl_tensor.strides) {
    stride_offset = snprintf(strides, 2, "[");
  } else {
    stride_offset = snprintf(strides, sizeof("NULL"), "NULL");
  }
  for (int i=0; i < dl_tensor->dl_tensor.ndim - 1; ++i) {
    shape_offset += snprintf(shape + shape_offset, 23, "%ld, ", dl_tensor->dl_tensor.shape[i]);
    if (dl_tensor->dl_tensor.strides) {
      stride_offset += snprintf(strides + stride_offset, 23, "%ld, ", dl_tensor->dl_tensor.strides[i]);
    }
  }
  if (dl_tensor->dl_tensor.strides) {
    snprintf(strides + stride_offset, 22, "%ld]",
	     dl_tensor->dl_tensor.strides[dl_tensor->dl_tensor.ndim-1]);
  }
  snprintf(shape + shape_offset, 22, "%ld]",
	   dl_tensor->dl_tensor.shape[dl_tensor->dl_tensor.ndim-1]);
  
  printf("struct DLManagedTensor {\n\tDLTensor dl_tensor == {\n"
	 "\t\tvoid* data == %p;\n\t\tDLContext ctx == {\n"
	 "\t\t\tDLDeviceType device_type == %d;\n\t\t\tint device_id == %d;\n"
	 "\t\t};\n\t\tint ndim == %d;\n\t\tDLDataType dtype == {\n"
	 "\t\t\tuint8_t code == %u;\n\t\t\tuint8_t bits == %u;\n"
	 "\t\t\tuint16_t lanes == %u;\n\t\t};\n\t\tint64_t *shape == %s;\n"
	 "\t\tint64_t *strides == %s\n\t\tuint64_t byte_offset == %lu;\n"
	 "\t};\n\t void * manager_ctx == %p;\n"
	 "\tvoid (*deleter)(struct DLManagedTensor * self) == %p\n"
	 "};\n",
	 dl_tensor->dl_tensor.data, dl_tensor->dl_tensor.ctx.device_type,
	 dl_tensor->dl_tensor.ctx.device_id, dl_tensor->dl_tensor.ndim,
	 dl_tensor->dl_tensor.dtype.code, dl_tensor->dl_tensor.dtype.bits,
	 dl_tensor->dl_tensor.dtype.lanes, shape, strides,
	 dl_tensor->dl_tensor.byte_offset,
	 dl_tensor->manager_ctx,
	 dl_tensor->deleter);
  return CEED_ERROR_SUCCESS;
}
