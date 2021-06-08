/// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
/// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
/// reserved. See files LICENSE and NOTICE for details.
///
/// This file is part of CEED, a collection of benchmarks, miniapps, software
/// libraries and APIs for efficient high-order finite element and spectral
/// element discretizations for exascale applications. For more information and
/// source code availability see http://github.com/ceed.
///
/// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
/// a collaborative effort of two U.S. Department of Energy organizations (Office
/// of Science and the National Nuclear Security Administration) responsible for
/// the planning and preparation of a capable exascale ecosystem, including
/// software, applications, hardware, advanced system engineering and early
/// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef _ceed_dlpack_h
#define _ceed_dlpack_h

#include <ceed/ceed.h>
#include <dlpack.h>

CEED_EXTERN int CeedVectorTakeFromDLPack(Ceed ceed,
					 CeedVector vec,
					 DLManagedTensor *dl_tensor,
					 CeedCopyMode copy_mode);

CEED_EXTERN int CeedVectorToDLPack(Ceed ceed,
				   CeedVector vec,
				   CeedMemType dl_mem_type,
				   DLManagedTensor **dl_tensor);

#endif
