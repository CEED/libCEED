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

#ifndef _ceed_fortran_name_h
#define _ceed_fortran_name_h

/* establishes some macros to establish
   * the FORTRAN naming convention
     default      gs_setup, etc.
     -DUPCASE     GS_SETUP, etc.
     -DUNDERSCORE gs_setup_, etc.
   * a prefix for all external (non-FORTRAN) function names
     for example, -DPREFIX=jl_   transforms fail -> jl_fail
   * a prefix for all external FORTRAN function names
     for example, -DFPREFIX=jlf_ transforms gs_setup_ -> jlf_gs_setup_
*/

/* the following macro functions like a##b,
   but will expand a and/or b if they are themselves macros */
#define TOKEN_PASTE_(a,b) a##b
#define TOKEN_PASTE(a,b) TOKEN_PASTE_(a,b)

#ifdef PREFIX
#  define PREFIXED_NAME(x) TOKEN_PASTE(PREFIX,x)
#else
#  define PREFIXED_NAME(x) x
#endif

#ifdef FPREFIX
#  define FPREFIXED_NAME(x) TOKEN_PASTE(FPREFIX,x)
#else
#  define FPREFIXED_NAME(x) x
#endif

#if defined(UPCASE)
#  define FORTRAN_NAME(low,up) FPREFIXED_NAME(up)
#  define FORTRAN_UNPREFIXED(low,up) up
#elif defined(UNDERSCORE)
#  define FORTRAN_NAME(low,up) FPREFIXED_NAME(TOKEN_PASTE(low,_))
#  define FORTRAN_UNPREFIXED(low,up) TOKEN_PASTE(low,_)
#else
#  define FORTRAN_NAME(low,up) FPREFIXED_NAME(low)
#  define FORTRAN_UNPREFIXED(low,up) low
#endif

#endif
