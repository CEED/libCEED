// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#define CEED_DEBUG_COLOR 178
#include "ceed-occa.h"

// *****************************************************************************
// * CeedOklPath_Occa
// *****************************************************************************
int CeedOklPath_Occa(const Ceed ceed, const char *c_src_file,
                     const char *okl_base_name, char **okl_file) {
  struct stat buf;
  const Ceed_Occa *ceed_data = ceed->data;
  int ierr = CeedCalloc(4096,okl_file); CeedChk(ierr);
  memcpy(*okl_file,c_src_file,strlen(c_src_file));
  char *okl = *okl_file;
  const char *last_dot = strrchr(okl,'.');
  if (!last_dot)
    return CeedError(ceed, 1, "Cannot find file's extension!");
  const size_t okl_path_len = last_dot - okl;
  strcpy(&okl[okl_path_len],".okl");
  dbg("[CeedOklPath] current OKL is %s",okl);
  // Test if we can get file's status,
  if (stat(okl, &buf)!=0) { // if not revert to occa cache
    dbg("[CeedOklPath] Could NOT stat this OKL file: %s",okl);
    dbg("[CeedOklPath] Reverting to occa://ceed/*.okl");
    // Try to stat ceed-occa-restrict.okl in occa cache
    ierr=sprintf(okl,"%s/libraries/ceed/%s.okl",
                 ceed_data->occa_cache_dir,
                 okl_base_name);
    CeedChk(ierr);
    // if we cannot find the okl file in cache,
    if (stat(okl, &buf)!=0) { // look into libceed install path
      dbg("[CeedOklPath] Could NOT stat OCCA cache: %s",okl);
      ierr=sprintf(okl,"%s/okl/%s.okl",
                   ceed_data->libceed_dir,
                   okl_base_name);
      CeedChk(ierr);
    } else // if it is in occa cache, use it
      sprintf(okl,"occa://ceed/%s.okl",okl_base_name);
  }
  dbg("[CeedOklPath]   final OKL is %s",okl);
  return 0;
}

// *****************************************************************************
// * CeedOklDladdr_Occa for Apple and Linux
// *****************************************************************************
int CeedOklDladdr_Occa(Ceed ceed) {
  Dl_info info;
  Ceed_Occa *data = ceed->data;
  memset(&info,0,sizeof(info));
  int ierr = dladdr((void*)&CeedInit,&info);
  dbg("[CeedOklDladdr] libceed -> %s", info.dli_fname);
  if (ierr==0)
    return CeedError(ceed, 1, "OCCA backend cannot fetch dladdr");
  // libceed_dir setup & alloc in our data
  const char *libceed_dir = info.dli_fname;
  const int path_len = strlen(libceed_dir);
  ierr = CeedCalloc(path_len+1,&data->libceed_dir); CeedChk(ierr);
  // stat'ing the library to make sure we can access the path
  struct stat buf;
  if (stat(libceed_dir, &buf)!=0)
    return CeedError(ceed, 1, "OCCA backend cannot stat %s", libceed_dir);
  // Now remove libceed.so part to get the path
  const char *last_slash = strrchr(libceed_dir,'/');
  if (!last_slash)
    return CeedError(ceed, 1,
                     "OCCA backend cannot locate libceed_dir from %s",
                     libceed_dir);
  // remove from last_slash and push it to our data
  memcpy(data->libceed_dir,libceed_dir,last_slash-libceed_dir);
  dbg("[CeedOklDladdr] libceed_dir -> %s", data->libceed_dir);
  return 0;
}
