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
NDEBUG ?= 1

LDFLAGS ?=
UNDERSCORE ?= 1

# OCCA_DIR env variable should point to OCCA master (github.com/libocca/occa)
OCCA_DIR ?= ../occa

# env variable MAGMA_DIR can be used too
MAGMA_DIR ?= ../magma
# If CUDA_DIR is not set, check for nvcc, or resort to /usr/local/cuda
CUDA_DIR  ?= $(or $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir \
               $(shell which nvcc 2> /dev/null))))),/usr/local/cuda)

# Warning: SANTIZ options still don't run with /gpu/occa
# export LSAN_OPTIONS=suppressions=.asanignore
AFLAGS = -fsanitize=address #-fsanitize=undefined -fno-omit-frame-pointer

CFLAGS = -std=c99 -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP
FFLAGS = -cpp     -Wall -Wextra -Wno-unused-parameter -Wno-unused-dummy-argument -fPIC -MMD -MP
# If using the IBM XL Fortran (xlf) replace FFLAGS appropriately:
ifneq ($(filter %xlf %xlf_r,$(FC)),)
  FFLAGS = -qpreprocess -qextname -qpic -MMD
endif

CFLAGS += $(if $(NDEBUG),-O2 -DNDEBUG=1,-g)

ifeq ($(UNDERSCORE), 1)
  CFLAGS += -DUNDERSCORE
endif

FFLAGS += $(if $(NDEBUG),-O2 -DNDEBUG=1,-g)

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
okldir = $(prefix)/lib/okl
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

PROVE ?= prove
PROVE_OPTS ?= -j $(NPROCS)
DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)

ceed.pc := $(LIBDIR)/pkgconfig/ceed.pc
libceed := $(LIBDIR)/libceed.$(SO_EXT)
libceed.c := $(wildcard ceed*.c)

# Tests
tests.c   := $(sort $(wildcard tests/t[0-9][0-9]-*.c))
tests.f   := $(sort $(wildcard tests/t[0-9][0-9]-*.f))
tests     := $(tests.c:tests/%.c=$(OBJDIR)/%)
ctests    := $(tests)
tests     += $(tests.f:tests/%.f=$(OBJDIR)/%)
#examples
examples.c := $(sort $(wildcard examples/ceed/*.c))
examples.f := $(sort $(wildcard examples/ceed/*.f))
examples  := $(examples.c:examples/ceed/%.c=$(OBJDIR)/%)
examples  += $(examples.f:examples/ceed/%.f=$(OBJDIR)/%)
# backends/[ref & occa  & magma & cuda]
ref.c     := $(sort $(wildcard backends/ref/*.c))
template.c:= $(sort $(wildcard backends/template/*.c))
occa.c    := $(sort $(wildcard backends/occa/*.c))
cuda.c    := $(sort $(wildcard backends/cuda/*.c))
cuda.cu   := $(sort $(wildcard backends/cuda/*.cu))
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

.SUFFIXES:
.SUFFIXES: .c .o .d
.SECONDEXPANSION: # to expand $$(@D)/.DIR

.SECONDARY: $(magma_tmp.c) $(magma_tmp.cu)

%/.DIR :
	@mkdir -p $(@D)
	@touch $@

.PRECIOUS: %/.DIR

this: $(libceed) $(ceed.pc)
# run 'this' target in parallel
all:;@$(MAKE) $(MFLAGS) V=$(V) this

$(libceed) : LDFLAGS += $(if $(DARWIN), -install_name @rpath/$(notdir $(libceed)))

libceed.c += $(ref.c)
libceed.c += $(template.c)
libceed.c += $(cuda.c)
libceed.cu += $(cuda.cu)
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  $(libceed) : LDFLAGS += -L$(OCCA_DIR)/lib -Wl,-rpath,$(abspath $(OCCA_DIR)/lib)
  $(libceed) : LDLIBS += -locca
  libceed.c += $(occa.c)
  $(occa.c:%.c=$(OBJDIR)/%.o) : CFLAGS += -I$(OCCA_DIR)/include
endif
ifneq ($(wildcard $(MAGMA_DIR)/lib/libmagma.*),)
  CUDA_LIB_DIR := $(wildcard $(foreach d,lib lib64,$(CUDA_DIR)/$d/libcudart.${SO_EXT}))
  CUDA_LIB_DIR := $(patsubst %/,%,$(dir $(firstword $(CUDA_LIB_DIR))))
  ifneq ($(CUDA_LIB_DIR),)
  cuda_link = -Wl,-rpath,$(CUDA_LIB_DIR) -L$(CUDA_LIB_DIR) -lcublas -lcusparse -lcudart
  omp_link = -fopenmp
  magma_link_static = -L$(MAGMA_DIR)/lib -lmagma $(cuda_link) $(omp_link)
  magma_link_shared = -L$(MAGMA_DIR)/lib -Wl,-rpath,$(abspath $(MAGMA_DIR)/lib) -lmagma
  magma_link := $(if $(wildcard $(MAGMA_DIR)/lib/libmagma.${SO_EXT}),$(magma_link_shared),$(magma_link_static))
  magma_allsrc.o = $(magma_allsrc.c:%.c=$(OBJDIR)/%.o) $(magma_allsrc.cu:%.cu=$(OBJDIR)/%.o)
  $(libceed)           : LDLIBS += $(magma_link)
  $(tests) $(examples) : LDLIBS += $(magma_link)
  $(libceed) : $(magma_allsrc.o)
  libceed.c  += $(magma_allsrc.c)
  libceed.cu += $(magma_allsrc.cu)
  $(magma_allsrc.c:%.c=$(OBJDIR)/%.o) : CFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
  $(magma_allsrc.cu:%.cu=$(OBJDIR)/%.o) : NVCCFLAGS += --compiler-options=-fPIC -DADD_ -I$(MAGMA_DIR)/include -I$(MAGMA_DIR)/magmablas -I$(MAGMA_DIR)/control -I$(CUDA_DIR)/include
  endif
endif

# generate magma_tmp.c and magma_cuda.cu from magma.c
$(magma_tmp.c) $(magma_tmp.cu): $(magma_pre_src) | $$(@D)/.DIR
	$(magma_preprocessor) $<

$(libceed) : $(libceed.c:%.c=$(OBJDIR)/%.o) $(libceed.cu:%.cu=$(OBJDIR)/%.o) | $$(@D)/.DIR
	$(call quiet,CC) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : %.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : %.cu | $$(@D)/.DIR
	$(call quiet,NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/% : tests/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : tests/%.f | $$(@D)/.DIR
	$(call quiet,FC) $(CPPFLAGS) $(FFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : examples/ceed/%.f | $$(@D)/.DIR
	$(call quiet,FC) $(CPPFLAGS) $(FFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(tests) $(examples) : $(libceed)
$(tests) $(examples) : LDFLAGS += -Wl,-rpath,$(abspath $(LIBDIR)) -L$(LIBDIR)

run-% : $(OBJDIR)/%
	@tests/tap.sh $(<:build/%=%)

test : $(tests:$(OBJDIR)/%=run-%) $(examples:$(OBJDIR)/%=run-%)
# run test target in parallel
tst : ;@$(MAKE) $(MFLAGS) V=$(V) test
# CPU C tests only for backend %
ctc-% : $(ctests);@$(foreach tst,$(ctests),$(tst) /cpu/$*;)

prove : $(tests) $(examples)
	$(PROVE) $(PROVE_OPTS) --exec 'tests/tap.sh' $(tests:$(OBJDIR)/%=%) $(examples:$(OBJDIR)/%=%)
# run prove target in parallel
prv : ;@$(MAKE) $(MFLAGS) V=$(V) prove

examples : $(examples)

$(ceed.pc) : pkgconfig-prefix = $(abspath .)
$(OBJDIR)/ceed.pc : pkgconfig-prefix = $(prefix)
.INTERMEDIATE : $(OBJDIR)/ceed.pc
%/ceed.pc : ceed.pc.template | $$(@D)/.DIR
	@sed "s:%prefix%:$(pkgconfig-prefix):" $< > $@

# The occa executable is not linked with RPATH by default, so we need it to find its libocca.so
OCCA               := $(if $(DARWIN),DYLD_LIBRARY_PATH,LD_LIBRARY_PATH)=$(OCCA_DIR)/lib $(OCCA_DIR)/bin/occa
OKL_KERNELS        := $(wildcard backends/occa/*.okl)

okl-cache :
	$(OCCA) cache ceed $(OKL_KERNELS)

okl-clear:
	$(OCCA) clear -y -l ceed

install : $(libceed) $(OBJDIR)/ceed.pc
	$(INSTALL) -d "$(DESTDIR)$(includedir)" "$(DESTDIR)$(libdir)" "$(DESTDIR)$(okldir)" "$(DESTDIR)$(pkgconfigdir)"
	$(INSTALL_DATA) include/ceed.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) include/ceedf.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) $(libceed) "$(DESTDIR)$(libdir)/"
	$(INSTALL_DATA) $(OBJDIR)/ceed.pc "$(DESTDIR)$(pkgconfigdir)/"
	$(INSTALL_DATA) $(OKL_KERNELS) "$(DESTDIR)$(okldir)/"

.PHONY : all cln clean print test tst prove prv examples style install doc okl-cache okl-clear

cln clean :
	$(RM) *.o *.d $(libceed)
	$(RM) -r *.dSYM $(OBJDIR) $(LIBDIR)/pkgconfig
	$(MAKE) -C examples/ceed clean
	$(MAKE) -C examples/mfem clean
	$(MAKE) -C examples/petsc clean
	(cd examples/nek5000 && bash make-nek-examples.sh clean)
	$(RM) $(magma_tmp.c) $(magma_tmp.cu) backends/magma/*~ backends/magma/*.o

distclean : clean
	rm -rf doc/html

doc :
	doxygen Doxyfile

style :
	astyle --style=google --indent=spaces=2 --max-code-length=80 \
            --keep-one-line-statements --keep-one-line-blocks --lineend=linux \
            --suffix=none --preserve-date --formatted \
            *.[ch] tests/*.[ch] backends/*/*.[ch] examples/*/*.[ch] examples/*/*.[ch]pp

print :
	@echo $(VAR)=$($(VAR))

print-% :
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

-include $(libceed.c:%.c=build/%.d) $(tests.c:tests/%.c=build/%.d)
