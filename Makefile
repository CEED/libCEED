# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

CONFIG ?= config.mk
-include $(CONFIG)
COMMON ?= common.mk
-include $(COMMON)

ifeq (,$(filter-out undefined default,$(origin CC)))
  CC = gcc
endif
ifeq (,$(filter-out undefined default,$(origin CXX)))
  CXX = g++
endif
ifeq (,$(filter-out undefined default,$(origin FC)))
  FC = gfortran
endif
ifeq (,$(filter-out undefined default,$(origin LINK)))
  LINK = $(CC)
endif
ifeq (,$(filter-out undefined default,$(origin AR)))
  AR = ar
endif
ifeq (,$(filter-out undefined default,$(origin ARFLAGS)))
  ARFLAGS = crD
endif
NVCC ?= $(CUDA_DIR)/bin/nvcc
NVCC_CXX ?= $(CXX)
HIPCC ?= $(HIP_DIR)/bin/hipcc
SED ?= sed

# ASAN must be left empty if you don't want to use it
ASAN ?=

LDFLAGS ?=
UNDERSCORE ?= 1

# MFEM_DIR env variable should point to sibling directory
ifneq ($(wildcard ../mfem/libmfem.*),)
  MFEM_DIR ?= ../mfem
endif

# NEK5K_DIR env variable should point to sibling directory
ifneq ($(wildcard ../Nek5000/*),)
  NEK5K_DIR ?= $(abspath ../Nek5000)
endif
export NEK5K_DIR
MPI ?= 1

# CEED_DIR env for NEK5K testing
export CEED_DIR = $(abspath .)

# XSMM_DIR env variable should point to XSMM master (github.com/hfp/libxsmm)
XSMM_DIR ?= ../libxsmm

# OCCA_DIR env variable should point to OCCA master (github.com/libocca/occa)
OCCA_DIR ?= ../occa

# env variable MAGMA_DIR can be used too
MAGMA_DIR ?= ../magma
# If CUDA_DIR is not set, check for nvcc, or resort to /usr/local/cuda
CUDA_DIR  ?= $(or $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir \
               $(shell which nvcc 2> /dev/null))))),/usr/local/cuda)
CUDA_ARCH ?= 
HIP_DIR ?= /opt/rocm
HIP_ARCH ?=

# Check for PETSc in ../petsc
ifneq ($(wildcard ../petsc/lib/libpetsc.*),)
  PETSC_DIR ?= ../petsc
endif

# Warning: SANTIZ options still don't run with /gpu/occa
# export LSAN_OPTIONS=suppressions=.asanignore
AFLAGS = -fsanitize=address #-fsanitize=undefined -fno-omit-frame-pointer

# Note: Intel oneAPI C/C++ compiler is now icx/icpx
CC_VENDOR := $(subst oneAPI,icc,$(patsubst gcc%,gcc,$(firstword $(filter gcc% clang icc oneAPI XL,$(shell $(CC) --version)))))
FC_VENDOR := $(firstword $(filter GNU ifort XL,$(shell $(FC) --version 2>&1 || $(FC) -qversion)))

# Default extra flags by vendor
MARCHFLAG.gcc           := -march=native
MARCHFLAG.clang         := $(MARCHFLAG.gcc)
MARCHFLAG.icc           :=
OMP_SIMD_FLAG.gcc       := -fopenmp-simd
OMP_SIMD_FLAG.clang     := $(OMP_SIMD_FLAG.gcc)
OMP_SIMD_FLAG.icc       := -qopenmp-simd
OPT.gcc                 := -ffp-contract=fast
OPT.clang               := $(OPT.gcc)
CFLAGS.gcc              := -fPIC -std=c99 -Wall -Wextra -Wno-unused-parameter -MMD -MP
CFLAGS.clang            := $(CFLAGS.gcc)
CFLAGS.icc              := $(CFLAGS.gcc)
CFLAGS.XL               := -qpic -MMD
CXXFLAGS.gcc            := -fPIC -std=c++11 -Wall -Wextra -Wno-unused-parameter -MMD -MP
CXXFLAGS.clang          := $(CXXFLAGS.gcc)
CXXFLAGS.icc            := $(CXXFLAGS.gcc)
CXXFLAGS.XL             := -qpic -std=c++11 -MMD
FFLAGS.GNU              := -fPIC -cpp -Wall -Wextra -Wno-unused-parameter -Wno-unused-dummy-argument -MMD -MP
FFLAGS.ifort            := -fPIC -cpp
FFLAGS.XL               := -qpic -ffree-form -qpreprocess -qextname -MMD

# This check works with compilers that use gcc and clang.  It fails with some
# compilers; e.g., xlc apparently ignores all options when -E is passed, thus
# succeeds with any flags.  Users can pass MARCHFLAG=... if desired.
cc_check_flag = $(shell $(CC) -E -Werror $(1) -x c /dev/null > /dev/null 2>&1 && echo 1)
MARCHFLAG := $(MARCHFLAG.$(CC_VENDOR))
MARCHFLAG := $(if $(call cc_check_flag,$(MARCHFLAG)),$(MARCHFLAG),-mcpu=native)
MARCHFLAG := $(if $(call cc_check_flag,$(MARCHFLAG)),$(MARCHFLAG))

OMP_SIMD_FLAG := $(OMP_SIMD_FLAG.$(CC_VENDOR))
OMP_SIMD_FLAG := $(if $(call cc_check_flag,$(OMP_SIMD_FLAG)),$(OMP_SIMD_FLAG))

OPT    ?= -O -g $(MARCHFLAG) $(OPT.$(CC_VENDOR)) $(OMP_SIMD_FLAG)
CFLAGS ?= $(OPT) $(CFLAGS.$(CC_VENDOR))
CXXFLAGS ?= $(OPT) $(CXXFLAGS.$(CC_VENDOR))
LIBCXX ?= -lstdc++
NVCCFLAGS ?= -ccbin $(CXX) -Xcompiler "$(OPT)" -Xcompiler -fPIC
ifneq ($(CUDA_ARCH),)
  NVCCFLAGS += -arch=$(CUDA_ARCH)
endif
HIPCCFLAGS ?= $(filter-out $(OMP_SIMD_FLAG),$(OPT)) -fPIC
ifneq ($(HIP_ARCH),)
  HIPCCFLAGS += --amdgpu-target=$(HIP_ARCH)
endif
FFLAGS ?= $(OPT) $(FFLAGS.$(FC_VENDOR))

ifeq ($(COVERAGE), 1)
  CFLAGS += --coverage
  CXXFLAGS += --coverage
  LDFLAGS += --coverage
endif

CFLAGS += $(if $(ASAN),$(AFLAGS))
FFLAGS += $(if $(ASAN),$(AFLAGS))
LDFLAGS += $(if $(ASAN),$(AFLAGS))
CPPFLAGS += -I./include
LDLIBS = -lm
OBJDIR := build
LIBDIR := lib

# Installation variables
prefix ?= /usr/local
bindir = $(prefix)/bin
libdir = $(prefix)/lib
includedir = $(prefix)/include
pkgconfigdir = $(libdir)/pkgconfig
INSTALL = install
INSTALL_PROGRAM = $(INSTALL)
INSTALL_DATA = $(INSTALL) -m644

# Get number of processors of the machine
NPROCS := $(shell getconf _NPROCESSORS_ONLN)
# prepare make options to run in parallel
MFLAGS := -j $(NPROCS) --warn-undefined-variables \
                       --no-print-directory --no-keep-going

PYTHON ?= python3
PROVE ?= prove
PROVE_OPTS ?= -j $(NPROCS)
DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)

ceed.pc := $(LIBDIR)/pkgconfig/ceed.pc
libceed.so := $(LIBDIR)/libceed.$(SO_EXT)
libceed.a := $(LIBDIR)/libceed.a
libceed := $(if $(STATIC),$(libceed.a),$(libceed.so))
CEED_LIBS = -lceed
libceed.c := $(filter-out interface/ceed-cuda.c interface/ceed-hip.c, $(wildcard interface/ceed*.c backends/*.c gallery/*.c))
gallery.c := $(wildcard gallery/*/ceed*.c)
libceed.c += $(gallery.c)
libceed_test.so := $(LIBDIR)/libceed_test.$(SO_EXT)
libceed_test.a := $(LIBDIR)/libceed_test.a
libceed_test := $(if $(STATIC),$(libceed_test.a),$(libceed_test.so))
libceeds = $(libceed) $(libceed_test)
BACKENDS_BUILTIN := /cpu/self/ref/serial /cpu/self/ref/blocked /cpu/self/opt/serial /cpu/self/opt/blocked
BACKENDS := $(BACKENDS_BUILTIN)

# Tests
tests.c   := $(sort $(wildcard tests/t[0-9][0-9][0-9]-*.c))
tests.f   := $(sort $(wildcard tests/t[0-9][0-9][0-9]-*.f90))
tests     := $(tests.c:tests/%.c=$(OBJDIR)/%)
ctests    := $(tests)
tests     += $(tests.f:tests/%.f90=$(OBJDIR)/%)
# Examples
examples.c := $(sort $(wildcard examples/ceed/*.c))
examples.f := $(sort $(wildcard examples/ceed/*.f))
examples  := $(examples.c:examples/ceed/%.c=$(OBJDIR)/%)
examples  += $(examples.f:examples/ceed/%.f=$(OBJDIR)/%)
# MFEM Examples
mfemexamples.cpp := $(sort $(wildcard examples/mfem/*.cpp))
mfemexamples  := $(mfemexamples.cpp:examples/mfem/%.cpp=$(OBJDIR)/mfem-%)
# Nek5K Examples
nekexamples  := $(OBJDIR)/nek-bps
# PETSc Examples
petscexamples.c := $(wildcard examples/petsc/*.c)
petscexamples   := $(petscexamples.c:examples/petsc/%.c=$(OBJDIR)/petsc-%)
# Fluid Dynamics Examples
fluidsexamples.c := $(sort $(wildcard examples/fluids/*.c))
fluidsexamples  := $(fluidsexamples.c:examples/fluids/%.c=$(OBJDIR)/fluids-%)
# Solid Mechanics Examples
solidsexamples.c := $(sort $(wildcard examples/solids/*.c))
solidsexamples   := $(solidsexamples.c:examples/solids/%.c=$(OBJDIR)/solids-%)

# Backends/[ref, blocked, template, memcheck, opt, avx, occa, magma]
ref.c          := $(sort $(wildcard backends/ref/*.c))
blocked.c      := $(sort $(wildcard backends/blocked/*.c))
template.c     := $(sort $(wildcard backends/template/*.c))
ceedmemcheck.c := $(sort $(wildcard backends/memcheck/*.c))
opt.c          := $(sort $(wildcard backends/opt/*.c))
avx.c          := $(sort $(wildcard backends/avx/*.c))
xsmm.c         := $(sort $(wildcard backends/xsmm/*.c))
cuda.c         := $(sort $(wildcard backends/cuda/*.c))
cuda.cpp       := $(sort $(wildcard backends/cuda/*.cpp))
cuda.cu        := $(sort $(wildcard backends/cuda/kernels/*.cu))
cuda-shared.c  := $(sort $(wildcard backends/cuda-shared/*.c))
cuda-shared.cu := $(sort $(wildcard backends/cuda-shared/kernels/*.cu))
cuda-gen.c     := $(sort $(wildcard backends/cuda-gen/*.c))
cuda-gen.cpp   := $(sort $(wildcard backends/cuda-gen/*.cpp))
cuda-gen.cu    := $(sort $(wildcard backends/cuda-gen/kernels/*.cu))
occa.cpp       := $(sort $(shell find backends/occa -type f -name *.cpp))
magma.c        := $(sort $(wildcard backends/magma/*.c))
magma.cu       := $(sort $(wildcard backends/magma/kernels/cuda/*.cu))
magma.hip      := $(sort $(wildcard backends/magma/kernels/hip/*.hip.cpp))
hip.c          := $(sort $(wildcard backends/hip/*.c))
hip.cpp        := $(sort $(wildcard backends/hip/*.cpp))
hip.hip        := $(sort $(wildcard backends/hip/kernels/*.hip.cpp))
hip-shared.c   := $(sort $(wildcard backends/hip-shared/*.c))
hip-gen.c      := $(sort $(wildcard backends/hip-gen/*.c))
hip-gen.cpp    := $(sort $(wildcard backends/hip-gen/*.cpp))

# Quiet, color output
quiet ?= $($(1))

# Cancel built-in and old-fashioned implicit rules which we don't use
.SUFFIXES:

.SECONDEXPANSION: # to expand $$(@D)/.DIR

%/.DIR :
	@mkdir -p $(@D)
	@touch $@

.PRECIOUS: %/.DIR

lib: $(libceed) $(ceed.pc)
# run 'lib' target in parallel
par:;@$(MAKE) $(MFLAGS) V=$(V) lib
backend_status = $(if $(filter $1,$(BACKENDS)), [backends: $1], [not found])
info:
	$(info ------------------------------------)
	$(info CC            = $(CC))
	$(info CXX           = $(CXX))
	$(info FC            = $(FC))
	$(info CPPFLAGS      = $(CPPFLAGS))
	$(info CFLAGS        = $(value CFLAGS))
	$(info CXXFLAGS      = $(value CXXFLAGS))
	$(info FFLAGS        = $(value FFLAGS))
	$(info NVCCFLAGS     = $(value NVCCFLAGS))
	$(info HIPCCFLAGS    = $(value HIPCCFLAGS))
	$(info LDFLAGS       = $(value LDFLAGS))
	$(info LDLIBS        = $(LDLIBS))
	$(info AR            = $(AR))
	$(info ARFLAGS       = $(ARFLAGS))
	$(info OPT           = $(OPT))
	$(info AFLAGS        = $(AFLAGS))
	$(info ASAN          = $(or $(ASAN),(empty)))
	$(info V             = $(or $(V),(empty)) [verbose=$(if $(V),on,off)])
	$(info ------------------------------------)
	$(info MEMCHK_STATUS = $(MEMCHK_STATUS)$(call backend_status,$(MEMCHK_BACKENDS)))
	$(info AVX_STATUS    = $(AVX_STATUS)$(call backend_status,$(AVX_BACKENDS)))
	$(info XSMM_DIR      = $(XSMM_DIR)$(call backend_status,$(XSMM_BACKENDS)))
	$(info OCCA_DIR      = $(OCCA_DIR)$(call backend_status,$(OCCA_BACKENDS)))
	$(info MAGMA_DIR     = $(MAGMA_DIR)$(call backend_status,$(MAGMA_BACKENDS)))
	$(info CUDA_DIR      = $(CUDA_DIR)$(call backend_status,$(CUDA_BACKENDS)))
	$(info HIP_DIR       = $(HIP_DIR)$(call backend_status,$(HIP_BACKENDS)))
	$(info ------------------------------------)
	$(info MFEM_DIR      = $(MFEM_DIR))
	$(info NEK5K_DIR     = $(NEK5K_DIR))
	$(info PETSC_DIR     = $(PETSC_DIR))
	$(info ------------------------------------)
	$(info prefix        = $(prefix))
	$(info includedir    = $(value includedir))
	$(info libdir        = $(value libdir))
	$(info pkgconfigdir  = $(value pkgconfigdir))
	$(info ------------------------------------)
	@true
info-backends:
	$(info make: 'lib' with optional backends: $(filter-out $(BACKENDS_BUILTIN),$(BACKENDS)))
.PHONY: lib all par info info-backends

$(libceed.so) : LDFLAGS += $(if $(DARWIN), -install_name @rpath/$(notdir $(libceed.so)))
$(libceed_test) : LDFLAGS += $(if $(DARWIN), -install_name @rpath/$(notdir $(libceed_test)))

# Standard Backends
libceed.c += $(ref.c)
libceed.c += $(blocked.c)
libceed.c += $(opt.c)

# Testing Backends
test_backends.c := $(template.c)
TEST_BACKENDS := /cpu/self/tmpl /cpu/self/tmpl/sub

# Memcheck Backend
MEMCHK_STATUS = Disabled
MEMCHK := $(shell echo "\#include <valgrind/memcheck.h>" | $(CC) $(CPPFLAGS) -E - >/dev/null 2>&1 && echo 1)
MEMCHK_BACKENDS = /cpu/self/memcheck/serial /cpu/self/memcheck/blocked
ifeq ($(MEMCHK),1)
  MEMCHK_STATUS = Enabled
  libceed.c += $(ceedmemcheck.c)
  BACKENDS += $(MEMCHK_BACKENDS)
endif

# AVX Backed
AVX_STATUS = Disabled
AVX_FLAG := $(if $(filter clang,$(CC_VENDOR)),+avx,-mavx)
AVX := $(filter $(AVX_FLAG),$(shell $(CC) $(OPT) -v -E -x c /dev/null 2>&1))
AVX_BACKENDS = /cpu/self/avx/serial /cpu/self/avx/blocked
ifneq ($(AVX),)
  AVX_STATUS = Enabled
  libceed.c += $(avx.c)
  BACKENDS += $(AVX_BACKENDS)
endif

# Collect list of libraries and paths for use in linking and pkg-config
PKG_LIBS =

# libXSMM Backends
XSMM_BACKENDS = /cpu/self/xsmm/serial /cpu/self/xsmm/blocked
ifneq ($(wildcard $(XSMM_DIR)/lib/libxsmm.*),)
  PKG_LIBS += -L$(abspath $(XSMM_DIR))/lib -lxsmm -ldl
  MKL ?=
  ifeq (,$(MKL)$(MKLROOT))
    BLAS_LIB = -lblas
  else
    ifneq ($(MKLROOT),)
      # Some installs put everything inside an intel64 subdirectory, others not
      MKL_LIBDIR = $(dir $(firstword $(wildcard $(MKLROOT)/lib/intel64/libmkl_sequential.* $(MKLROOT)/lib/libmkl_sequential.*)))
      MKL_LINK = -L$(MKL_LIBDIR)
      PKG_LIB_DIRS += $(MKL_LIBDIR)
    endif
    BLAS_LIB = $(MKL_LINK) -Wl,--push-state,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -Wl,--pop-state
  endif
  PKG_LIBS += $(BLAS_LIB)
  libceed.c += $(xsmm.c)
  $(xsmm.c:%.c=$(OBJDIR)/%.o) $(xsmm.c:%=%.tidy) : CPPFLAGS += -I$(XSMM_DIR)/include
  BACKENDS += $(XSMM_BACKENDS)
endif

# OCCA Backends
OCCA_BACKENDS = /cpu/self/occa
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  OCCA_MODES := $(shell $(OCCA_DIR)/bin/occa modes)
  OCCA_BACKENDS += $(if $(filter OpenMP,$(OCCA_MODES)),/cpu/openmp/occa)
# OCCA_BACKENDS += $(if $(filter OpenCL,$(OCCA_MODES)),/gpu/opencl/occa)
  OCCA_BACKENDS += $(if $(filter HIP,$(OCCA_MODES)),/gpu/hip/occa)
  OCCA_BACKENDS += $(if $(filter CUDA,$(OCCA_MODES)),/gpu/cuda/occa)

  $(libceeds) : CPPFLAGS += -I$(OCCA_DIR)/include
  PKG_LIBS += -L$(abspath $(OCCA_DIR))/lib -locca
  LIBCEED_CONTAINS_CXX = 1
  libceed.cpp += $(occa.cpp)
  BACKENDS += $(OCCA_BACKENDS)
endif

# CUDA Backends
CUDA_LIB_DIR := $(wildcard $(foreach d,lib lib64,$(CUDA_DIR)/$d/libcudart.${SO_EXT}))
CUDA_LIB_DIR := $(patsubst %/,%,$(dir $(firstword $(CUDA_LIB_DIR))))
CUDA_LIB_DIR_STUBS := $(CUDA_LIB_DIR)/stubs
CUDA_BACKENDS = /gpu/cuda/ref /gpu/cuda/shared /gpu/cuda/gen
ifneq ($(CUDA_LIB_DIR),)
  $(libceeds) : CPPFLAGS += -I$(CUDA_DIR)/include
  PKG_LIBS += -L$(abspath $(CUDA_LIB_DIR)) -lcudart -lnvrtc -lcuda -lcublas
  LIBCEED_CONTAINS_CXX = 1
  libceed.c   += interface/ceed-cuda.c
  libceed.c   += $(cuda.c) $(cuda-shared.c) $(cuda-gen.c)
  libceed.cpp += $(cuda.cpp) $(cuda-gen.cpp)
  libceed.cu  += $(cuda.cu) $(cuda-shared.cu) $(cuda-gen.cu)
  BACKENDS    += $(CUDA_BACKENDS)
endif

# HIP Backends
HIP_LIB_DIR := $(wildcard $(foreach d,lib lib64,$(HIP_DIR)/$d/libamdhip64.${SO_EXT}))
HIP_LIB_DIR := $(patsubst %/,%,$(dir $(firstword $(HIP_LIB_DIR))))
HIP_BACKENDS = /gpu/hip/ref /gpu/hip/shared /gpu/hip/gen
ifneq ($(HIP_LIB_DIR),)
  $(libceeds) : HIPCCFLAGS += -I./include
  ifneq ($(CXX), $(HIPCC))
    CPPFLAGS += $(subst =,,$(shell $(HIP_DIR)/bin/hipconfig -C))
  endif
  $(libceeds) : CPPFLAGS += -I$(HIP_DIR)/include
  PKG_LIBS += -L$(abspath $(HIP_LIB_DIR)) -lamdhip64 -lhipblas
  LIBCEED_CONTAINS_CXX = 1
  libceed.c   += interface/ceed-hip.c
  libceed.c   += $(hip.c) $(hip-shared.c) $(hip-gen.c)
  libceed.cpp += $(hip.cpp) $(hip-gen.cpp)
  libceed.hip += $(hip.hip)
  BACKENDS    += $(HIP_BACKENDS)
endif

# MAGMA Backend
ifneq ($(wildcard $(MAGMA_DIR)/lib/libmagma.*),)
  MAGMA_ARCH=$(shell nm -g $(MAGMA_DIR)/lib/libmagma.* | grep -c "hipblas")
  ifeq ($(MAGMA_ARCH), 0) #CUDA MAGMA
    ifneq ($(CUDA_LIB_DIR),)
      cuda_link = -Wl,-rpath,$(CUDA_LIB_DIR) -L$(CUDA_LIB_DIR) -lcublas -lcusparse -lcudart
      omp_link = -fopenmp
      magma_link_static = -L$(MAGMA_DIR)/lib -lmagma $(cuda_link) $(omp_link)
      magma_link_shared = -L$(MAGMA_DIR)/lib -Wl,-rpath,$(abspath $(MAGMA_DIR)/lib) -lmagma
      magma_link := $(if $(wildcard $(MAGMA_DIR)/lib/libmagma.${SO_EXT}),$(magma_link_shared),$(magma_link_static))
      PKG_LIBS += $(magma_link)
      libceed.c  += $(magma.c)
      libceed.cu += $(magma.cu)
      $(magma.c:%.c=$(OBJDIR)/%.o) $(magma.c:%=%.tidy) : CPPFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
      $(magma.cu:%.cu=$(OBJDIR)/%.o) : CPPFLAGS += --compiler-options=-fPIC -DADD_ -I$(MAGMA_DIR)/include -I$(MAGMA_DIR)/magmablas -I$(MAGMA_DIR)/control -I$(CUDA_DIR)/include
      MAGMA_BACKENDS = /gpu/cuda/magma /gpu/cuda/magma/det
    endif
  else  # HIP MAGMA
    ifneq ($(HIP_LIB_DIR),)
      hip_link = -Wl,-rpath,$(HIP_LIB_DIR) -L$(HIP_LIB_DIR) -lhipblas -lhipsparse -lamdhip64
      omp_link = -fopenmp
      magma_link_static = -L$(MAGMA_DIR)/lib -lmagma $(hip_link) $(omp_link)
      magma_link_shared = -L$(MAGMA_DIR)/lib -Wl,-rpath,$(abspath $(MAGMA_DIR)/lib) -lmagma
      magma_link := $(if $(wildcard $(MAGMA_DIR)/lib/libmagma.${SO_EXT}),$(magma_link_shared),$(magma_link_static))
      PKG_LIBS += $(magma_link)
      libceed.c  += $(magma.c)
      libceed.hip += $(magma.hip)
      ifneq ($(CXX), $(HIPCC))
        $(magma.c:%.c=$(OBJDIR)/%.o) $(magma.c:%=%.tidy) : CPPFLAGS += -I$(MAGMA_DIR)/include -I$(HIP_DIR)/include -DHAVE_HIP -DADD_
      else
        $(magma.c:%.c=$(OBJDIR)/%.o) $(magma.c:%=%.tidy) : HIPCCFLAGS += -I$(MAGMA_DIR)/include -I$(HIP_DIR)/include -DHAVE_HIP -DADD_
      endif
      $(magma.hip:%.hip.cpp=$(OBJDIR)/%.o) : HIPCCFLAGS += -I$(MAGMA_DIR)/include -I$(MAGMA_DIR)/magmablas -I$(MAGMA_DIR)/control -I$(HIP_DIR)/include -DHAVE_HIP -DADD_
      MAGMA_BACKENDS = /gpu/hip/magma /gpu/hip/magma/det
    endif
  endif
  LIBCEED_CONTAINS_CXX = 1
  BACKENDS += $(MAGMA_BACKENDS)
endif

export BACKENDS

_pkg_ldflags = $(filter -L%,$(PKG_LIBS))
_pkg_ldlibs = $(filter-out -L%,$(PKG_LIBS))
$(libceeds) : LDFLAGS += $(_pkg_ldflags) $(_pkg_ldflags:-L%=-Wl,-rpath,%)
$(libceeds) : LDLIBS += $(_pkg_ldlibs)
ifeq ($(STATIC),1)
$(examples) $(tests) : LDFLAGS += $(_pkg_ldflags) $(_pkg_ldflags:-L%=-Wl,-rpath,%)
$(examples) $(tests) : LDLIBS += $(_pkg_ldlibs)
endif

pkgconfig-libs-private = $(PKG_LIBS)
ifeq ($(LIBCEED_CONTAINS_CXX),1)
  $(libceeds) : LINK = $(CXX)
  ifeq ($(STATIC),1)
    $(examples) $(tests) : LDLIBS += $(LIBCXX)
	  pkgconfig-libs-private += $(LIBCXX)
  endif
endif

# File names *-weak.c contain weak symbol definitions, which must be listed last
# when creating shared or static libraries.
weak_last = $(filter-out %-weak.o,$(1)) $(filter %-weak.o,$(1))

libceed.o = $(libceed.c:%.c=$(OBJDIR)/%.o) $(libceed.cpp:%.cpp=$(OBJDIR)/%.o) $(libceed.cu:%.cu=$(OBJDIR)/%.o) $(libceed.hip:%.hip.cpp=$(OBJDIR)/%.o)
$(filter %fortran.o,$(libceed.o)) : CPPFLAGS += $(if $(filter 1,$(UNDERSCORE)),-DUNDERSCORE)
$(libceed.o): | info-backends
$(libceed.so) : $(call weak_last,$(libceed.o)) | $$(@D)/.DIR
	$(call quiet,LINK) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(libceed.a) : $(call weak_last,$(libceed.o)) | $$(@D)/.DIR
	$(call quiet,AR) $(ARFLAGS) $@ $^

$(OBJDIR)/%.o : $(CURDIR)/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : $(CURDIR)/%.cpp | $$(@D)/.DIR
	$(call quiet,CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : $(CURDIR)/%.cu | $$(@D)/.DIR
	$(call quiet,NVCC) $(filter-out -Wno-unused-function, $(CPPFLAGS)) $(NVCCFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : $(CURDIR)/%.hip.cpp | $$(@D)/.DIR
	$(call quiet,HIPCC) $(HIPCCFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/% : tests/%.c | $$(@D)/.DIR
	$(call quiet,LINK.c) $(CEED_LDFLAGS) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/% : tests/%.f90 | $$(@D)/.DIR
	$(call quiet,LINK.F) -DSOURCE_DIR='"$(abspath $(<D))/"' $(CEED_LDFLAGS) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.c | $$(@D)/.DIR
	$(call quiet,LINK.c) $(CEED_LDFLAGS) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.f | $$(@D)/.DIR
	$(call quiet,LINK.F) -DSOURCE_DIR='"$(abspath $(<D))/"' $(CEED_LDFLAGS) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/mfem-% : examples/mfem/%.cpp $(libceed) | $$(@D)/.DIR
	+$(MAKE) -C examples/mfem CEED_DIR=`pwd` \
	  MFEM_DIR="$(abspath $(MFEM_DIR))" CXX=$(CXX) $*
	mv examples/mfem/$* $@

# Note: Multiple Nek files cannot be built in parallel. The '+' here enables
#       this single Nek bps file to be built in parallel with other examples,
#       such as when calling `make prove-all -j2`.
$(OBJDIR)/nek-bps : examples/nek/bps/bps.usr examples/nek/nek-examples.sh $(libceed) | $$(@D)/.DIR
	+$(MAKE) -C examples MPI=$(MPI) CEED_DIR=`pwd` NEK5K_DIR="$(abspath $(NEK5K_DIR))" nek
	mv examples/nek/build/bps $(OBJDIR)/bps
	cp examples/nek/nek-examples.sh $(OBJDIR)/nek-bps

$(OBJDIR)/petsc-% : examples/petsc/%.c $(libceed) $(ceed.pc) | $$(@D)/.DIR
	+$(MAKE) -C examples/petsc CEED_DIR=`pwd` \
	  PETSC_DIR="$(abspath $(PETSC_DIR))" OPT="$(OPT)" $*
	mv examples/petsc/$* $@

$(OBJDIR)/fluids-% : examples/fluids/%.c $(libceed) $(ceed.pc) | $$(@D)/.DIR
	+$(MAKE) -C examples/fluids CEED_DIR=`pwd` \
	  PETSC_DIR="$(abspath $(PETSC_DIR))" OPT="$(OPT)" $*
	mv examples/fluids/$* $@

$(OBJDIR)/solids-% : examples/solids/%.c $(libceed) $(ceed.pc) | $$(@D)/.DIR
	+$(MAKE) -C examples/solids CEED_DIR=`pwd` \
	  PETSC_DIR="$(abspath $(PETSC_DIR))" OPT="$(OPT)" $*
	mv examples/solids/$* $@

libceed_test.o = $(test_backends.c:%.c=$(OBJDIR)/%.o)
$(libceed_test.so) : $(call weak_last,$(libceed.o) $(libceed_test.o)) | $$(@D)/.DIR
	$(call quiet,LINK) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(libceed_test.a) : $(call weak_last,$(libceed.o) $(libceed_test.o)) | $$(@D)/.DIR
	$(call quiet,AR) $(ARFLAGS) $@ $^

$(examples) : $(libceed)
$(tests) : $(libceed_test)
$(tests) : CEED_LIBS = -lceed_test
$(tests) $(examples) : LDFLAGS += -Wl,-rpath,$(abspath $(LIBDIR)) -L$(LIBDIR)

run-t% : BACKENDS += $(TEST_BACKENDS)
run-% : $(OBJDIR)/%
	@tests/tap.sh $(<:$(OBJDIR)/%=%)

external_examples := \
	$(if $(MFEM_DIR),$(mfemexamples)) \
	$(if $(PETSC_DIR),$(petscexamples)) \
	$(if $(NEK5K_DIR),$(nekexamples)) \
	$(if $(PETSC_DIR),$(fluidsexamples)) \
	$(if $(PETSC_DIR),$(solidsexamples))

allexamples = $(examples) $(external_examples)

# The test and prove targets can be controlled via pattern searches.  The
# default is to run tests and those examples that have no external dependencies.
# Examples of finer grained control:
#
#   make test search='petsc mfem'      # PETSc and MFEM examples
#   make prove search='t3'             # t3xx series tests
#   make junit search='ex petsc'       # core ex* and PETSc tests
search ?= t ex
realsearch = $(search:%=%%)
matched = $(foreach pattern,$(realsearch),$(filter $(OBJDIR)/$(pattern),$(tests) $(allexamples)))

# Test core libCEED
test : $(matched:$(OBJDIR)/%=run-%)

# Run test target in parallel
tst : ;@$(MAKE) $(MFLAGS) V=$(V) test
# CPU C tests only for backend %
ctc-% : $(ctests);@$(foreach tst,$(ctests),$(tst) /cpu/$*;)

prove : BACKENDS += $(TEST_BACKENDS)
prove : $(matched)
	$(info Testing backends: $(BACKENDS))
	$(PROVE) $(PROVE_OPTS) --exec 'tests/tap.sh' $(matched:$(OBJDIR)/%=%)
# Run prove target in parallel
prv : ;@$(MAKE) $(MFLAGS) V=$(V) prove

prove-all :
	+$(MAKE) prove realsearch=%

junit-t% : BACKENDS += $(TEST_BACKENDS)
junit-% : $(OBJDIR)/%
	@printf "  %10s %s\n" TEST $(<:$(OBJDIR)/%=%); $(PYTHON) tests/junit.py $(<:$(OBJDIR)/%=%)

junit : $(matched:$(OBJDIR)/%=junit-%)

all: $(alltests)

examples : $(allexamples)
ceedexamples : $(examples)
nekexamples : $(nekexamples)
mfemexamples : $(mfemexamples)
petscexamples : $(petscexamples)

# Benchmarks
allbenchmarks = petsc-bps
bench_targets = $(addprefix bench-,$(allbenchmarks))
.PHONY: $(bench_targets) benchmarks
$(bench_targets): bench-%: $(OBJDIR)/%
	cd benchmarks && ./benchmark.sh --ceed "$(BACKENDS)" -r $(*).sh
benchmarks: $(bench_targets)

$(ceed.pc) : pkgconfig-prefix = $(abspath .)
$(OBJDIR)/ceed.pc : pkgconfig-prefix = $(prefix)
.INTERMEDIATE : $(OBJDIR)/ceed.pc
%/ceed.pc : ceed.pc.template | $$(@D)/.DIR
	@$(SED) \
	    -e "s:%prefix%:$(pkgconfig-prefix):" \
	    -e "s:%libs_private%:$(pkgconfig-libs-private):" $< > $@

install : $(libceed) $(OBJDIR)/ceed.pc
	$(INSTALL) -d $(addprefix $(if $(DESTDIR),"$(DESTDIR)"),"$(includedir)"\
	  "$(includedir)/ceed/" "$(libdir)" "$(pkgconfigdir)")
	$(INSTALL_DATA) include/ceed/ceed.h "$(DESTDIR)$(includedir)/ceed/"
	$(INSTALL_DATA) include/ceed/fortran.h "$(DESTDIR)$(includedir)/ceed/"
	$(INSTALL_DATA) include/ceed/backend.h "$(DESTDIR)$(includedir)/ceed/"
	$(INSTALL_DATA) include/ceed/cuda.h "$(DESTDIR)$(includedir)/ceed/"
	$(INSTALL_DATA) include/ceed/hip.h "$(DESTDIR)$(includedir)/ceed/"
	$(INSTALL_DATA) include/ceed/hash.h "$(DESTDIR)$(includedir)/ceed/"
	$(INSTALL_DATA) include/ceed/khash.h "$(DESTDIR)$(includedir)/ceed/"
	$(INSTALL_DATA) $(libceed) "$(DESTDIR)$(libdir)/"
	$(INSTALL_DATA) $(OBJDIR)/ceed.pc "$(DESTDIR)$(pkgconfigdir)/"
	$(INSTALL_DATA) include/ceed.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) include/ceedf.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) include/ceed-backend.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) include/ceed-hash.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) include/ceed-khash.h "$(DESTDIR)$(includedir)/"

.PHONY : cln clean doxygen doc lib install all print test tst prove prv prove-all junit examples style style-c style-py tidy info info-backends

cln clean :
	$(RM) -r $(OBJDIR) $(LIBDIR) dist *egg* .pytest_cache *cffi*
	$(MAKE) -C examples clean NEK5K_DIR="$(abspath $(NEK5K_DIR))"
	$(MAKE) -C tests/python clean
	$(RM) benchmarks/*output.txt

distclean : clean
	$(RM) -r doc/html doc/sphinx/build $(CONFIG)

DOXYGEN ?= doxygen
doxygen :
	$(DOXYGEN) Doxyfile

doc-html doc-latexpdf doc-epub doc-livehtml : doc-% : doxygen
	make -C doc/sphinx $*

doc : doc-html

style-c :
	@astyle --options=.astylerc \
          $(filter-out include/ceedf.h $(wildcard tests/t*-f.h), \
            $(wildcard include/*.h interface/*.[ch] tests/*.[ch] backends/*/*.[ch] \
              examples/*/*/*.[ch] examples/*/*.[ch] examples/*/*.[ch]pp gallery/*/*.[ch]))

AUTOPEP8 = autopep8
style-py : AUTOPEP8_ARGS = --in-place --aggressive
style-py :
	@$(AUTOPEP8) $(AUTOPEP8_ARGS) $(wildcard *.py python**/*.py tests/python**/*.py examples**/*.py doc/sphinx/source**/*.py benchmarks/*.py)

style : style-c style-py

CLANG_TIDY ?= clang-tidy

%.c.tidy : %.c
	$(CLANG_TIDY) $(TIDY_OPTS) $^ -- $(CPPFLAGS) --std=c99 -I$(CUDA_DIR)/include -I$(HIP_DIR)/include

%.cpp.tidy : %.cpp
	$(CLANG_TIDY) $(TIDY_OPTS) $^ -- $(CPPFLAGS) --std=c++11 -I$(CUDA_DIR)/include -I$(OCCA_DIR)/include -I$(HIP_DIR)/include

tidy_c   : $(libceed.c:%=%.tidy)
tidy_cpp : $(libceed.cpp:%=%.tidy)

tidy : tidy_c tidy_cpp

print :
	@echo $(VAR)=$($(VAR))

print-% :
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [        flavor]: $(flavor $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

# "make configure" detects any variables passed on the command line or
# previously set in config.mk, caching them in config.mk as simple
# (:=) variables.  Variables set in config.mk or on the command line
# take precedence over the defaults provided in the file.  Typical
# usage:
#
#   make configure CC=/path/to/my/cc CUDA_DIR=/opt/cuda
#   make
#   make prove
#
# The values in the file can be updated by passing them on the command
# line, e.g.,
#
#   make configure CC=/path/to/other/clang

# All variables to consider for caching
CONFIG_VARS = CC CXX FC NVCC NVCC_CXX HIPCC \
	OPT CFLAGS CPPFLAGS CXXFLAGS FFLAGS NVCCFLAGS HIPCCFLAGS \
	AR ARFLAGS LDFLAGS LDLIBS LIBCXX SED \
	MAGMA_DIR XSMM_DIR CUDA_DIR CUDA_ARCH MFEM_DIR PETSC_DIR NEK5K_DIR HIP_DIR HIP_ARCH

# $(call needs_save,CFLAGS) returns true (a nonempty string) if CFLAGS
# was set on the command line or in config.mk (where it will appear as
# a simple variable).
needs_save = $(or $(filter command line,$(origin $(1))),$(filter simple,$(flavor $(1))))

configure :
	$(file > $(CONFIG))
	$(foreach v,$(CONFIG_VARS),$(if $(call needs_save,$(v)),$(file >> $(CONFIG),$(v) := $($(v)))))
	@echo "Configuration cached in $(CONFIG):"
	@cat $(CONFIG)

wheel : export MARCHFLAG = -march=generic
wheel : export WHEEL_PLAT = manylinux2010_x86_64
wheel :
	docker run -it --user $(shell id -u):$(shell id -g) --rm -v $(PWD):/io -w /io \
		-e MARCHFLAG -e WHEEL_PLAT \
		quay.io/pypa/$(WHEEL_PLAT) python/make-wheels.sh

.PHONY : configure wheel

# Include *.d deps when not -B = --always-make: useful if the paths are wonky in a container
-include $(if $(filter B,$(MAKEFLAGS)),,$(libceed.c:%.c=$(OBJDIR)/%.d) $(tests.c:tests/%.c=$(OBJDIR)/%.d))
