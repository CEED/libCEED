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

ifeq (,$(filter-out undefined default,$(origin CC)))
  CC = gcc
endif
ifeq (,$(filter-out undefined default,$(origin FC)))
  FC = gfortran
endif
NVCC = $(CUDA_DIR)/bin/nvcc

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
  NEK5K_DIR ?= ../Nek5000
endif

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

OPT    = -O -g -march=native -ffp-contract=fast
CFLAGS = -std=c99 $(OPT) -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP
NVCCFLAGS = -Xcompiler "$(OPT)" -Xcompiler -fPIC
# If using the IBM XL Fortran (xlf) replace FFLAGS appropriately:
ifneq ($(filter %xlf %xlf_r,$(FC)),)
  FFLAGS = $(OPT) -ffree-form -qpreprocess -qextname -qpic -MMD
else # gfortran/Intel-style options
  FFLAGS = -cpp     $(OPT) -Wall -Wextra -Wno-unused-parameter -Wno-unused-dummy-argument -fPIC -MMD -MP
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
libceed.c := $(wildcard interface/ceed*.c)
BACKENDS_BUILTIN := /cpu/self/ref/serial /cpu/self/ref/blocked /cpu/self/tmpl
BACKENDS := $(BACKENDS_BUILTIN)

# Tests
tests.c   := $(sort $(wildcard tests/t[0-9][0-9][0-9]-*.c))
tests.f   := $(sort $(wildcard tests/t[0-9][0-9][0-9]-*.f90))
tests     := $(tests.c:tests/%.c=$(OBJDIR)/%)
ctests    := $(tests)
tests     += $(tests.f:tests/%.f90=$(OBJDIR)/%)
#examples
examples.c := $(sort $(wildcard examples/ceed/*.c))
examples.f := $(sort $(wildcard examples/ceed/*.f))
examples  := $(examples.c:examples/ceed/%.c=$(OBJDIR)/%)
examples  += $(examples.f:examples/ceed/%.f=$(OBJDIR)/%)
#mfemexamples
mfemexamples.cpp := $(sort $(wildcard examples/mfem/*.cpp))
mfemexamples  := $(mfemexamples.cpp:examples/mfem/%.cpp=$(OBJDIR)/mfem-%)
#nekexamples
nekexamples.usr := $(sort $(wildcard examples/nek5000/*.usr))
nekexamples  := $(nekexamples.usr:examples/nek5000/%.usr=nek-%)
#petscexamples
petscexamples.c := $(sort $(wildcard examples/petsc/*.c))
petscexamples  := $(petscexamples.c:examples/petsc/%.c=$(OBJDIR)/petsc-%)

# backends/[ref, template, blocked, avx, occa, magma]
ref.c      := $(sort $(wildcard backends/ref/*.c))
template.c := $(sort $(wildcard backends/template/*.c))
cuda.c     := $(sort $(wildcard backends/cuda/*.c))
cuda.cu    := $(sort $(wildcard backends/cuda/*.cu))
cuda-reg.c := $(sort $(wildcard backends/cuda-reg/*.c))
cuda-reg.cu:= $(sort $(wildcard backends/cuda-reg/*.cu))
blocked.c  := $(sort $(wildcard backends/blocked/*.c))
avx.c      := $(sort $(wildcard backends/avx/*.c))
xsmm.c     := $(sort $(wildcard backends/xsmm/*.c))
occa.c     := $(sort $(wildcard backends/occa/*.c))
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
	$(info CC         = $(CC))
	$(info FC         = $(FC))
	$(info CPPFLAGS   = $(CPPFLAGS))
	$(info CFLAGS     = $(value CFLAGS))
	$(info FFLAGS     = $(value FFLAGS))
	$(info NVCCFLAGS  = $(value NVCCFLAGS))
	$(info LDFLAGS    = $(value LDFLAGS))
	$(info LDLIBS     = $(LDLIBS))
	$(info OPT        = $(OPT))
	$(info AFLAGS     = $(AFLAGS))
	$(info ASAN       = $(or $(ASAN),(empty)))
	$(info V          = $(or $(V),(empty)) [verbose=$(if $(V),on,off)])
	$(info ------------------------------------)
	$(info AVX_STATUS = $(AVX_STATUS)$(call backend_status,/cpu/self/avx/serial /cpu/self/avx/blocked))
	$(info XSMM_DIR   = $(XSMM_DIR)$(call backend_status,/cpu/self/xsmm/serial /cpu/self/xsmm/blocked))
	$(info OCCA_DIR   = $(OCCA_DIR)$(call backend_status,/cpu/occa /gpu/occa /omp/occa))
	$(info MAGMA_DIR  = $(MAGMA_DIR)$(call backend_status,/gpu/magma))
	$(info CUDA_DIR   = $(CUDA_DIR)$(call backend_status,/gpu/magma))
	$(info ------------------------------------)
	$(info MFEM_DIR   = $(MFEM_DIR))
	$(info NEK5K_DIR  = $(NEK5K_DIR))
	$(info PETSC_DIR  = $(PETSC_DIR))
	$(info ------------------------------------)
	$(info prefix       = $(prefix))
	$(info includedir   = $(value includedir))
	$(info libdir       = $(value libdir))
	$(info okldir       = $(value okldir))
	$(info pkgconfigdir = $(value pkgconfigdir))
	$(info ------------------------------------)
	@true
info-backends:
	$(info make: 'lib' with optional backends: $(filter-out $(BACKENDS_BUILTIN),$(BACKENDS)))
.PHONY: lib all par info info-backends

$(libceed) : LDFLAGS += $(if $(DARWIN), -install_name @rpath/$(notdir $(libceed)))

# Standard Backends
libceed.c += $(ref.c)
libceed.c += $(template.c)
libceed.c += $(blocked.c)

# AVX Backed
AVX_STATUS = Disabled
AVX := $(shell $(CC) $(OPT) -v -E - < /dev/null 2>&1 | grep -c avx)
ifeq ($(AVX),1)
  AVX_STATUS = Enabled
  libceed.c += $(avx.c)
  BACKENDS += /cpu/self/avx/serial /cpu/self/avx/blocked
endif

# libXSMM Backends
ifneq ($(wildcard $(XSMM_DIR)/lib/libxsmm.*),)
  $(libceed) : LDFLAGS += -L$(XSMM_DIR)/lib -Wl,-rpath,$(abspath $(XSMM_DIR)/lib)
  $(libceed) : LDLIBS += -lxsmm -ldl -lblas
  libceed.c += $(xsmm.c)
  $(xsmm.c:%.c=$(OBJDIR)/%.o) : CFLAGS += -I$(XSMM_DIR)/include
  BACKENDS += /cpu/self/xsmm/serial /cpu/self/xsmm/blocked
endif

# OCCA Backends
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  $(libceed) : LDFLAGS += -L$(OCCA_DIR)/lib -Wl,-rpath,$(abspath $(OCCA_DIR)/lib)
  $(libceed) : LDLIBS += -locca
  libceed.c += $(occa.c)
  $(occa.c:%.c=$(OBJDIR)/%.o) : CFLAGS += -I$(OCCA_DIR)/include
  BACKENDS += /cpu/occa /gpu/occa /omp/occa
endif

# Cuda Backend
CUDA_LIB_DIR := $(wildcard $(foreach d,lib lib64,$(CUDA_DIR)/$d/libcudart.${SO_EXT}))
CUDA_LIB_DIR := $(patsubst %/,%,$(dir $(firstword $(CUDA_LIB_DIR))))
ifneq ($(CUDA_LIB_DIR),)
  $(libceed) : CFLAGS += -I$(CUDA_DIR)/include
  $(libceed) : LDFLAGS += -L$(CUDA_LIB_DIR) -Wl,-rpath,$(abspath $(CUDA_LIB_DIR))
  $(libceed) : LDLIBS += -lcudart -lnvrtc -lcuda
  libceed.c  += $(cuda.c) $(cuda-reg.c)
  libceed.cu += $(cuda.cu) $(cuda-reg.cu)
  BACKENDS += /gpu/cuda/ref /gpu/cuda/reg
endif

# MAGMA Backend
ifneq ($(wildcard $(MAGMA_DIR)/lib/libmagma.*),)
  ifneq ($(CUDA_LIB_DIR),)
  cuda_link = -Wl,-rpath,$(CUDA_LIB_DIR) -L$(CUDA_LIB_DIR) -lcublas -lcusparse -lcudart
  omp_link = -fopenmp
  magma_link_static = -L$(MAGMA_DIR)/lib -lmagma $(cuda_link) $(omp_link)
  magma_link_shared = -L$(MAGMA_DIR)/lib -Wl,-rpath,$(abspath $(MAGMA_DIR)/lib) -lmagma
  magma_link := $(if $(wildcard $(MAGMA_DIR)/lib/libmagma.${SO_EXT}),$(magma_link_shared),$(magma_link_static))
  $(libceed)           : LDLIBS += $(magma_link)
  $(tests) $(examples) : LDLIBS += $(magma_link)
  libceed.c  += $(magma_allsrc.c)
  libceed.cu += $(magma_allsrc.cu)
  $(magma_allsrc.c:%.c=$(OBJDIR)/%.o) : CFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
  $(magma_allsrc.cu:%.cu=$(OBJDIR)/%.o) : NVCCFLAGS += --compiler-options=-fPIC -DADD_ -I$(MAGMA_DIR)/include -I$(MAGMA_DIR)/magmablas -I$(MAGMA_DIR)/control -I$(CUDA_DIR)/include
  BACKENDS += /gpu/magma
  endif
endif

export BACKENDS

# generate magma_tmp.c and magma_cuda.cu from magma.c
%_tmp.c %_cuda.cu : %.c
	$(magma_preprocessor) $<

libceed.o = $(libceed.c:%.c=$(OBJDIR)/%.o) $(libceed.cu:%.cu=$(OBJDIR)/%.o)
$(libceed.o): | info-backends
$(libceed) : $(libceed.o) | $$(@D)/.DIR
	$(call quiet,CC) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : $(CURDIR)/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : $(CURDIR)/%.cu | $$(@D)/.DIR
	$(call quiet,NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/% : tests/%.c | $$(@D)/.DIR
	$(call quiet,LINK.c) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : tests/%.f90 | $$(@D)/.DIR
	$(call quiet,LINK.F) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.c | $$(@D)/.DIR
	$(call quiet,LINK.c) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.f | $$(@D)/.DIR
	$(call quiet,LINK.F) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/mfem-% : examples/mfem/%.cpp $(libceed) | $$(@D)/.DIR
	$(MAKE) -C examples/mfem CEED_DIR=`pwd` \
	  MFEM_DIR="$(abspath $(MFEM_DIR))" $*
	mv examples/mfem/$* $@

$(OBJDIR)/petsc-% : examples/petsc/%.c $(libceed) $(ceed.pc) | $$(@D)/.DIR
	$(MAKE) -C examples/petsc CEED_DIR=`pwd` \
	  PETSC_DIR="$(abspath $(PETSC_DIR))" $*
	mv examples/petsc/$* $@

$(tests) $(examples) : $(libceed)
$(tests) $(examples) : LDFLAGS += -Wl,-rpath,$(abspath $(LIBDIR)) -L$(LIBDIR)

run-% : $(OBJDIR)/%
	@tests/tap.sh $(<:$(OBJDIR)/%=%)
# Test core libCEED
test : $(tests:$(OBJDIR)/%=run-%) $(examples:$(OBJDIR)/%=run-%)

# run test target in parallel
tst : ;@$(MAKE) $(MFLAGS) V=$(V) test
# CPU C tests only for backend %
ctc-% : $(ctests);@$(foreach tst,$(ctests),$(tst) /cpu/$*;)

prove : $(tests) $(examples)
	$(info Testing backends: $(BACKENDS))
	$(PROVE) $(PROVE_OPTS) --exec 'tests/tap.sh' $(tests:$(OBJDIR)/%=%) $(examples:$(OBJDIR)/%=%)
# run prove target in parallel
prv : ;@$(MAKE) $(MFLAGS) V=$(V) prove

allexamples := $(examples) $(if $(MFEM_DIR),$(mfemexamples)) $(if $(PETSC_DIR),$(petscexamples))
alltests := $(tests) $(allexamples)
fulltestlist = $(alltests) $(if $(NEK5K_DIR), $(nekexamples))
prepnektests:
	(export CC FC && cd examples && make prepnektests)
prove-all : $(alltests) $(if $(NEK5K_DIR), prepnektests)
	$(info Testing backends: $(BACKENDS))
	$(PROVE) $(PROVE_OPTS) --exec 'tests/tap.sh' $(fulltestlist:$(OBJDIR)/%=%)

junit-% : $(OBJDIR)/%
	@printf "  %10s %s\n" TEST $(<:$(OBJDIR)/%=%); $(PYTHON) tests/junit.py $(<:$(OBJDIR)/%=%)

junit : $(alltests:$(OBJDIR)/%=junit-%)

all: $(alltests)

examples : $(allexamples)

# Benchmarks
allbenchmarks = petsc-bp1 petsc-bp3
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

.PHONY : cln clean doc lib install all print test tst prove prv prove-all junit examples style okl-cache okl-clear info info-backends

cln clean :
	$(RM) -r $(OBJDIR) $(LIBDIR)
	$(MAKE) -C examples/ceed clean
	$(MAKE) -C examples/mfem clean
	$(MAKE) -C examples/petsc clean
	(cd examples/nek5000 && bash make-nek-examples.sh clean)
	$(RM) $(magma_tmp.c) $(magma_tmp.cu) backends/magma/*~ backends/magma/*.o
	$(RM) -rf benchmarks/*output.txt

distclean : clean
	$(RM) -r doc/html

doc :
	doxygen Doxyfile

style :
	@astyle --options=.astylerc \
          $(filter-out include/ceedf.h tests/t310-basis-f.h, \
            $(wildcard include/*.h interface/*.[ch] tests/*.[ch] backends/*/*.[ch] \
              examples/*/*.[ch] examples/*/*.[ch]pp))

print :
	@echo $(VAR)=$($(VAR))

print-% :
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

-include $(libceed.c:%.c=$(OBJDIR)/%.d) $(tests.c:tests/%.c=$(OBJDIR)/%.d)
