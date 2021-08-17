from libc.stdint cimport uintptr_t


cdef extern from "Python.h":
    ctypedef void (*PyCapsule_Destructor)(object)
    bint PyCapsule_IsValid(object, const char*)
    void* PyCapsule_GetPointer(object, const char*) except? NULL
    int PyCapsule_SetName(object, const char*) except -1
    object PyCapsule_New(void*, const char*, PyCapsule_Destructor)
    int PyCapsule_CheckExact(object)


cdef extern from "stdlib.h" nogil:
   ctypedef signed long int64_t
   ctypedef unsigned long long uint64_t
   ctypedef unsigned char uint8_t
   ctypedef unsigned short uint16_t
   void free(void* ptr)
   void* malloc(size_t size)


cdef extern from "dlpack.h":
   struct DLManagedTensor:
      pass








  



def DLPackPointerToCapsule(int ptr):
    cdef DLManagedTensor* dl_tensor = <DLManagedTensor*>(<uintptr_t>ptr)
    return PyCapsule_New(dl_tensor, 'dltensor', NULL)


def CapsuleToDLPackPointerValue(object capsule):
    cdef DLManagedTensor* ptr = <DLManagedTensor*>PyCapsule_GetPointer(capsule, 'dltensor')
    PyCapsule_SetName(capsule, 'used_dltensor') # make the data un-retrievable for future use 
    return <uintptr_t>ptr
