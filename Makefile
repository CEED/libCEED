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

-include config.mk

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
NVCC ?= $(CUDA_DIR)/bin/nvcc
NVCC_CXX ?= $(CXX)

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

# Check for PETSc in ../petsc
ifneq ($(wildcard ../petsc/lib/libpetsc.*),)
  PETSC_DIR ?= ../petsc
endif

# Warning: SANTIZ options still don't run with /gpu/occa
# export LSAN_OPTIONS=suppressions=.asanignore
AFLAGS = -fsanitize=address #-fsanitize=undefined -fno-omit-frame-pointer

OPT    = -O -g -march=native -ffp-contract=fast -fopenmp-simd
CFLAGS = -std=c99 $(OPT) -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP
CXXFLAGS = $(OPT) -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP
NVCCFLAGS = -ccbin $(CXX) -Xcompiler "$(OPT)" -Xcompiler -fPIC
# If using the IBM XL Fortran (xlf) replace FFLAGS appropriately:
ifneq ($(filter %xlf %xlf_r,$(FC)),)
  FFLAGS = $(OPT) -ffree-form -qpreprocess -qextname -qpic -MMD -DSOURCE_DIR='"$(abspath $(<D))/"'
else # gfortran/Intel-style options
  FFLAGS = -cpp     $(OPT) -Wall -Wextra -Wno-unused-parameter -Wno-unused-dummy-argument -fPIC -MMD -MP -DSOURCE_DIR='"$(abspath $(<D))/"'
endif

ifeq ($(UNDERSCORE), 1)
  CFLAGS += -DUNDERSCORE
endif

ifeq ($(COVERAGE), 1)
  CFLAGS += --coverage
  LDFLAGS += --coverage
endif

CFLAGS += $(if $(ASAN),$(AFLAGS))
FFLAGS += $(if $(ASAN),$(AFLAGS))
LDFLAGS += $(if $(ASAN),$(AFLAGS))
CPPFLAGS = -I./include
LDLIBS = -lm
OBJDIR := build
LIBDIR := lib

# Installation variables
prefix ?= /usr/local
bindir = $(prefix)/bin
libdir = $(prefix)/lib
okldir = $(libdir)/okl
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
libceed := $(LIBDIR)/libceed.$(SO_EXT)
CEED_LIBS = -lceed
libceed.c := $(wildcard interface/ceed*.c)
gallery.c := $(wildcard gallery/*/ceed*.c)
libceed.c += $(gallery.c)
libceed_test := $(LIBDIR)/libceed_test.$(SO_EXT)
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
# Navier-Stokes Example
navierstokesexample.c := $(sort $(wildcard examples/navier-stokes/*.c))
navierstokesexample  := $(navierstokesexample.c:examples/navier-stokes/%.c=$(OBJDIR)/navier-stokes-%)

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
cuda.cu        := $(sort $(wildcard backends/cuda/*.cu))
cuda-reg.c     := $(sort $(wildcard backends/cuda-reg/*.c))
cuda-reg.cu    := $(sort $(wildcard backends/cuda-reg/*.cu))
cuda-shared.c  := $(sort $(wildcard backends/cuda-shared/*.c))
cuda-shared.cu := $(sort $(wildcard backends/cuda-shared/*.cu))
cuda-gen.c     := $(sort $(wildcard backends/cuda-gen/*.c))
cuda-gen.cpp   := $(sort $(wildcard backends/cuda-gen/*.cpp))
cuda-gen.cu    := $(sort $(wildcard backends/cuda-gen/*.cu))
occa.c         := $(sort $(wildcard backends/occa/*.c))
magma_preprocessor := python backends/magma/gccm.py
magma_pre_src  := $(filter-out %_tmp.c, $(wildcard backends/magma/ceed-*.c))
magma_dsrc     := $(wildcard backends/magma/magma_d*.c)
magma_tmp.c    := $(magma_pre_src:%.c=%_tmp.c)
magma_tmp.cu   := $(magma_pre_src:%.c=%_cuda.cu)
magma_allsrc.c := $(magma_dsrc) $(magma_tmp.c)
magma_allsrc.cu:= $(magma_tmp.cu)

# Output using the 216-color rules mode
rule_file = $(notdir $(1))
rule_path = $(patsubst %/,%,$(dir $(1)))
last_path = $(notdir $(patsubst %/,%,$(dir $(1))))
ansicolor = $(shell echo $(call last_path,$(1)) | cksum | cut -b1-2 | xargs -IS expr 2 \* S + 17)
emacs_out = @printf "  %10s %s/%s\n" $(1) $(call rule_path,$(2)) $(call rule_file,$(2))
color_out = @if [ -t 1 ]; then \
				printf "  %10s \033[38;5;%d;1m%s\033[m/%s\n" \
					$(1) $(call ansicolor,$(2)) \
					$(call rule_path,$(2)) $(call rule_file,$(2)); else \
				printf "  %10s %s\n" $(1) $(2); fi
# if TERM=dumb, use it, otherwise switch to the term one
output = $(if $(TERM:dumb=),$(call color_out,$1,$2),$(call emacs_out,$1,$2))

# if V is set to non-nil, turn the verbose mode
quiet = $(if $(V),$($(1)),$(call output,$1,$@);$($(1)))

# Cancel built-in and old-fashioned implicit rules which we don't use
.SUFFIXES:

.SECONDEXPANSION: # to expand $$(@D)/.DIR

.SECONDARY: $(magma_tmp.c) $(magma_tmp.cu)

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
	$(info LDFLAGS       = $(value LDFLAGS))
	$(info LDLIBS        = $(LDLIBS))
	$(info OPT           = $(OPT))
	$(info AFLAGS        = $(AFLAGS))
	$(info ASAN          = $(or $(ASAN),(empty)))
	$(info V             = $(or $(V),(empty)) [verbose=$(if $(V),on,off)])
	$(info ------------------------------------)
	$(info MEMCHK_STATUS = $(MEMCHK_STATUS)$(call backend_status,/cpu/self/ref/memcheck))
	$(info AVX_STATUS    = $(AVX_STATUS)$(call backend_status,/cpu/self/avx/serial /cpu/self/avx/blocked))
	$(info XSMM_DIR      = $(XSMM_DIR)$(call backend_status,/cpu/self/xsmm/serial /cpu/self/xsmm/blocked))
	$(info OCCA_DIR      = $(OCCA_DIR)$(call backend_status,/cpu/occa /gpu/occa /omp/occa))
	$(info MAGMA_DIR     = $(MAGMA_DIR)$(call backend_status,/gpu/magma))
	$(info CUDA_DIR      = $(CUDA_DIR)$(call backend_status,$(CUDA_BACKENDS)))
	$(info ------------------------------------)
	$(info MFEM_DIR      = $(MFEM_DIR))
	$(info NEK5K_DIR     = $(NEK5K_DIR))
	$(info PETSC_DIR     = $(PETSC_DIR))
	$(info ------------------------------------)
	$(info prefix        = $(prefix))
	$(info includedir    = $(value includedir))
	$(info libdir        = $(value libdir))
	$(info okldir        = $(value okldir))
	$(info pkgconfigdir  = $(value pkgconfigdir))
	$(info ------------------------------------)
	@true
info-backends:
	$(info make: 'lib' with optional backends: $(filter-out $(BACKENDS_BUILTIN),$(BACKENDS)))
.PHONY: lib all par info info-backends

$(libceed) : LDFLAGS += $(if $(DARWIN), -install_name @rpath/$(notdir $(libceed)))
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
ifeq ($(MEMCHK),1)
  MEMCHK_STATUS = Enabled
  libceed.c += $(ceedmemcheck.c)
  BACKENDS += /cpu/self/ref/memcheck
endif

# AVX Backed
AVX_STATUS = Disabled
AVX := $(shell $(CC) $(OPT) -v -E - < /dev/null 2>&1 | grep -c ' -mavx')
ifeq ($(AVX),1)
  AVX_STATUS = Enabled
  libceed.c += $(avx.c)
  BACKENDS += /cpu/self/avx/serial /cpu/self/avx/blocked
endif

# libXSMM Backends
ifneq ($(wildcard $(XSMM_DIR)/lib/libxsmm.*),)
  $(libceeds) : LDFLAGS += -L$(XSMM_DIR)/lib -Wl,-rpath,$(abspath $(XSMM_DIR)/lib)
  $(libceeds) : LDLIBS += -lxsmm -ldl
  MKL ?=
  ifeq (,$(MKL)$(MKLROOT))
    BLAS_LIB = -lblas
  else
    BLAS_LIB = $(if $(MKLROOT),-L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64) -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
  endif
  $(libceeds) : LDLIBS += $(BLAS_LIB)
  libceed.c += $(xsmm.c)
  $(xsmm.c:%.c=$(OBJDIR)/%.o) $(xsmm.c:%=%.tidy) : CPPFLAGS += -I$(XSMM_DIR)/include
  BACKENDS += /cpu/self/xsmm/serial /cpu/self/xsmm/blocked
endif

# OCCA Backends
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  $(libceeds) : LDFLAGS += -L$(OCCA_DIR)/lib -Wl,-rpath,$(abspath $(OCCA_DIR)/lib)
  $(libceeds) : LDLIBS += -locca
  libceed.c += $(occa.c)
  $(occa.c:%.c=$(OBJDIR)/%.o) $(occa.c:%=%.tidy) : CPPFLAGS += -I$(OCCA_DIR)/include
  BACKENDS += /cpu/occa /gpu/occa /omp/occa
endif

# CUDA Backends
CUDA_LIB_DIR := $(wildcard $(foreach d,lib lib64,$(CUDA_DIR)/$d/libcudart.${SO_EXT}))
CUDA_LIB_DIR := $(patsubst %/,%,$(dir $(firstword $(CUDA_LIB_DIR))))
CUDA_LIB_DIR_STUBS := $(CUDA_LIB_DIR)/stubs
CUDA_BACKENDS = /gpu/cuda/ref /gpu/cuda/reg /gpu/cuda/shared /gpu/cuda/gen
ifneq ($(CUDA_LIB_DIR),)
  $(libceeds) : CFLAGS += -I$(CUDA_DIR)/include
  $(libceeds) : CPPFLAGS += -I$(CUDA_DIR)/include
  $(libceeds) : LDFLAGS += -L$(CUDA_LIB_DIR) -Wl,-rpath,$(abspath $(CUDA_LIB_DIR))
  $(libceeds) : LDLIBS += -lcudart -lnvrtc -lcuda
  $(libceeds) : LINK = $(CXX)
  libceed.c   += $(cuda.c) $(cuda-reg.c) $(cuda-shared.c) $(cuda-gen.c)
  libceed.cpp += $(cuda.cpp) $(cuda-gen.cpp)
  libceed.cu  += $(cuda.cu) $(cuda-reg.cu) $(cuda-shared.cu) $(cuda-gen.cu)
  BACKENDS += $(CUDA_BACKENDS)
endif

# MAGMA Backend
ifneq ($(wildcard $(MAGMA_DIR)/lib/libmagma.*),)
  ifneq ($(CUDA_LIB_DIR),)
  cuda_link = -Wl,-rpath,$(CUDA_LIB_DIR) -L$(CUDA_LIB_DIR) -lcublas -lcusparse -lcudart
  omp_link = -fopenmp
  magma_link_static = -L$(MAGMA_DIR)/lib -lmagma $(cuda_link) $(omp_link)
  magma_link_shared = -L$(MAGMA_DIR)/lib -Wl,-rpath,$(abspath $(MAGMA_DIR)/lib) -lmagma
  magma_link := $(if $(wildcard $(MAGMA_DIR)/lib/libmagma.${SO_EXT}),$(magma_link_shared),$(magma_link_static))
  $(libceeds)           : LDLIBS += $(magma_link)
  $(tests) $(examples) : LDLIBS += $(magma_link)
  libceed.c  += $(magma_allsrc.c)
  libceed.cu += $(magma_allsrc.cu)
  $(magma_allsrc.c:%.c=$(OBJDIR)/%.o) $(magma_allsrc.c:%=%.tidy) : CPPFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
  $(magma_allsrc.cu:%.cu=$(OBJDIR)/%.o) : NVCCFLAGS += --compiler-options=-fPIC -DADD_ -I$(MAGMA_DIR)/include -I$(MAGMA_DIR)/magmablas -I$(MAGMA_DIR)/control -I$(CUDA_DIR)/include
  BACKENDS += /gpu/magma
  endif
endif

export BACKENDS

# Generate magma_tmp.c and magma_cuda.cu from magma.c
%_tmp.c %_cuda.cu : %.c
	$(magma_preprocessor) $<

libceed.o = $(libceed.c:%.c=$(OBJDIR)/%.o) $(libceed.cpp:%.cpp=$(OBJDIR)/%.o) $(libceed.cu:%.cu=$(OBJDIR)/%.o)
$(libceed.o): | info-backends
$(libceed) : $(libceed.o) | $$(@D)/.DIR
	$(call quiet,LINK) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : $(CURDIR)/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : $(CURDIR)/%.cpp | $$(@D)/.DIR
	$(call quiet,CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : $(CURDIR)/%.cu | $$(@D)/.DIR
	$(call quiet,NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/% : tests/%.c | $$(@D)/.DIR
	$(call quiet,LINK.c) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/% : tests/%.f90 | $$(@D)/.DIR
	$(call quiet,LINK.F) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.c | $$(@D)/.DIR
	$(call quiet,LINK.c) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.f | $$(@D)/.DIR
	$(call quiet,LINK.F) -o $@ $(abspath $<) $(CEED_LIBS) $(LDLIBS)

$(OBJDIR)/mfem-% : examples/mfem/%.cpp $(libceed) | $$(@D)/.DIR
	+$(MAKE) -C examples/mfem CEED_DIR=`pwd` \
	  MFEM_DIR="$(abspath $(MFEM_DIR))" $*
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
	  PETSC_DIR="$(abspath $(PETSC_DIR))" $*
	mv examples/petsc/$* $@

$(OBJDIR)/navier-stokes-% : examples/navier-stokes/%.c $(libceed) $(ceed.pc) | $$(@D)/.DIR
	+$(MAKE) -C examples/navier-stokes CEED_DIR=`pwd` \
	  PETSC_DIR="$(abspath $(PETSC_DIR))" $*
	mv examples/navier-stokes/$* $@

libceed_test.o = $(test_backends.c:%.c=$(OBJDIR)/%.o)
$(libceed_test) : $(libceed.o) $(libceed_test.o) | $$(@D)/.DIR
	$(call quiet,LINK) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

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
	$(if $(NEK5K_DIR),$(nekexamples))

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
	@sed "s:%prefix%:$(pkgconfig-prefix):" $< > $@

OCCA        := $(OCCA_DIR)/bin/occa
OKL_KERNELS := $(wildcard backends/occa/*.okl)

okl-cache :
	$(OCCA) cache ceed $(OKL_KERNELS)

okl-clear:
	$(OCCA) clear -y -l ceed

install : $(libceed) $(OBJDIR)/ceed.pc
	$(INSTALL) -d $(addprefix $(if $(DESTDIR),"$(DESTDIR)"),"$(includedir)"\
	  "$(libdir)" "$(pkgconfigdir)" $(if $(OCCA_ON),"$(okldir)"))
	$(INSTALL_DATA) include/ceed.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) include/ceedf.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) $(libceed) "$(DESTDIR)$(libdir)/"
	$(INSTALL_DATA) $(OBJDIR)/ceed.pc "$(DESTDIR)$(pkgconfigdir)/"
	$(if $(OCCA_ON),$(INSTALL_DATA) $(OKL_KERNELS) "$(DESTDIR)$(okldir)/")

.PHONY : cln clean doc lib install all print test tst prove prv prove-all junit examples style tidy okl-cache okl-clear info info-backends

cln clean :
	$(RM) -r $(OBJDIR) $(LIBDIR)
	$(MAKE) -C examples clean NEK5K_DIR="$(abspath $(NEK5K_DIR))"
	$(RM) $(magma_tmp.c) $(magma_tmp.cu) backends/magma/*~ backends/magma/*.o
	$(RM) benchmarks/*output.txt

distclean : clean
	$(RM) -r doc/html config.mk

doc :
	doxygen Doxyfile

style :
	@astyle --options=.astylerc \
          $(filter-out include/ceedf.h tests/t310-basis-f.h, \
            $(wildcard include/*.h interface/*.[ch] tests/*.[ch] backends/*/*.[ch] \
              examples/*/*.[ch] examples/*/*.[ch]pp gallery/*/*.[ch]))

CLANG_TIDY ?= clang-tidy
%.c.tidy : %.c
	$(CLANG_TIDY) $^ -- $(CPPFLAGS)

tidy : $(libceed.c:%=%.tidy)

print :
	@echo $(VAR)=$($(VAR))

print-% :
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

# "make configure" will autodetect any variables not passed on the
# command line, caching the result in config.mk to be used on any
# subsequent invocations of make.  For example,
#
#   make configure CC=/path/to/my/cc CUDA_DIR=/opt/cuda
#   make
#   make prove
configure :
	@: > config.mk
	@echo "CC = $(CC)" | tee -a config.mk
	@echo "CXX = $(CXX)" | tee -a config.mk
	@echo "FC = $(FC)" | tee -a config.mk
	@echo "NVCC = $(NVCC)" | tee -a config.mk
	@echo "NVCC_CXX = $(NVCC_CXX)" | tee -a config.mk
	@echo "CFLAGS = $(CFLAGS)" | tee -a config.mk
	@echo "CPPFLAGS = $(CPPFLAGS)" | tee -a config.mk
	@echo "FFLAGS = $(FFLAGS)" | tee -a config.mk
	@echo "NVCCFLAGS = $(NVCCFLAGS)" | tee -a config.mk
	@echo "LDFLAGS = $(LDFLAGS)" | tee -a config.mk
	@echo "LDLIBS = $(LDLIBS)" | tee -a config.mk
	@echo "MAGMA_DIR = $(MAGMA_DIR)" | tee -a config.mk
	@echo "XSMM_DIR = $(XSMM_DIR)" | tee -a config.mk
	@echo "CUDA_DIR = $(CUDA_DIR)" | tee -a config.mk
	@echo "MFEM_DIR = $(MFEM_DIR)" | tee -a config.mk
	@echo "PETSC_DIR = $(PETSC_DIR)" | tee -a config.mk
	@echo "NEK5K_DIR = $(NEK5K_DIR)" | tee -a config.mk
	@echo "Configuration cached in config.mk"

.PHONY : configure

-include $(libceed.c:%.c=$(OBJDIR)/%.d) $(tests.c:tests/%.c=$(OBJDIR)/%.d)
