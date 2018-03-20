# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
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

CC  ?= gcc
FC   = gfortran
NVCC = nvcc

NDEBUG ?=
LDFLAGS ?=
LOADLIBES ?=
TARGET_ARCH ?=
UNDERSCORE ?= 1

# env variable OCCA_DIR should point to OCCA-1.0 branch
OCCA_DIR ?= ../occa

# env variable MAGMA_DIR should point to MAGMA branch
MAGMA_DIR ?= ../../magma
CUDA_DIR  ?= /usr/local/cuda

SANTIZ = -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
CFLAGS = -std=c99 -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP -march=native
FFLAGS = -cpp     -Wall -Wextra -Wno-unused-parameter -Wno-unused-dummy-argument -fPIC -MMD -MP -march=native

CFLAGS += $(if $(NDEBUG),-O2,-g)
ifeq ($(UNDERSCORE), 1)
  CFLAGS += -DUNDERSCORE
endif

FFLAGS += $(if $(NDEBUG),-O2,-g)

CFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
FFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
LDFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
CPPFLAGS = -I./include
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

NPROCS := $(shell getconf _NPROCESSORS_ONLN)
MFLAGS := -j $(NPROCS) --warn-undefined-variables \
			--no-print-directory --no-keep-going

PROVE ?= prove
PROVE_OPTS ?= -j $(NPROCS)
DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)

ceed.pc := $(LIBDIR)/pkgconfig/ceed.pc
libceed := $(LIBDIR)/libceed.$(SO_EXT)
libceed.c := $(wildcard ceed*.c)
# tests
tests.c   := $(sort $(wildcard tests/t[0-9][0-9]-*.c))
tests.f   := $(sort $(wildcard tests/t[0-9][0-9]-*.f))
tests     := $(tests.c:tests/%.c=$(OBJDIR)/%)
tests     += $(tests.f:tests/%.f=$(OBJDIR)/%)
#examples
examples.c := $(sort $(wildcard examples/*.c))
examples.f := $(sort $(wildcard examples/*.f))
examples  := $(examples.c:examples/%.c=$(OBJDIR)/%)
examples  += $(examples.f:examples/%.f=$(OBJDIR)/%)
# backends/[ref & occa & magma]
ref.c     := $(sort $(wildcard backends/ref/*.c))
occa.c    := $(sort $(wildcard backends/occa/*.c))
magma_preprocessor := python backends/magma/gccm.py
magma_pre_src := $(filter-out %_tmp.c, $(wildcard backends/magma/*.c))
magma.c       := $(magma_pre_src:%.c=%_tmp.c)
magma.cu      := $(magma_pre_src:%.c=%_cuda.cu)

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
.SECONDEXPANSION:		# to expand $$(@D)/.DIR

.SECONDARY: 

%/.DIR :
	@mkdir -p $(@D)
	@touch $@

.PRECIOUS: %/.DIR

this: $(libceed) $(ceed.pc)
all:;@$(MAKE) $(MFLAGS) V=$(V) this

$(libceed) : LDFLAGS += $(if $(DARWIN), -install_name $(abspath $(libceed)))

libceed.c += $(ref.c)
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  $(libceed) : LDFLAGS += -L$(OCCA_DIR)/lib -Wl,-rpath,$(abspath $(OCCA_DIR)/lib)
  $(libceed) : LDLIBS += -locca #-lrt -ldl
  libceed.c += $(occa.c)
  $(occa.c:%.c=$(OBJDIR)/%.o) : CFLAGS += -I$(OCCA_DIR)/include
endif
ifneq ($(wildcard $(MAGMA_DIR)/lib/libmagma.*),)
  $(libceed) : LDFLAGS += -L$(MAGMA_DIR)/lib -Wl,-rpath,$(abspath $(MAGMA_DIR)/lib)
  $(libceed) : LDLIBS += -lmagma -L$(CUDA_DIR)/lib -lcudart
  $(libceed) : $(magma.o)
  libceed.c += $(magma.c) 
  libceed.cu += $(magma.cu)
  $(magma.c:%.c=$(OBJDIR)/%.o) : CFLAGS += -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
  $(magma.cu:%.cu=$(OBJDIR)/%.o) : NVCCFLAGS += -I$(MAGMA_DIR)/include -I$(MAGMA_DIR)/magmablas -I$(MAGMA_DIR)/control -I$(CUDA_DIR)/include
endif

# generate magma_tmp.c and magma_cuda.cu from magma.c                                                      
%_tmp.c %_cuda.cu: %.c | $$(@D)/.DIR
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

$(OBJDIR)/% : examples/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : examples/%.f | $$(@D)/.DIR
	$(call quiet,FC) $(CPPFLAGS) $(FFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(tests) $(examples) : $(libceed)
$(tests) $(examples) : LDFLAGS += -Wl,-rpath,$(abspath $(LIBDIR)) -L$(LIBDIR)

run-% : $(OBJDIR)/%
	@tests/tap.sh $(<:build/%=%)

test : $(tests:$(OBJDIR)/%=run-%) $(examples:$(OBJDIR)/%=run-%)

prove : $(tests) $(examples)
	$(PROVE) $(PROVE_OPTS) --exec 'tests/tap.sh' $(tests:$(OBJDIR)/%=%) $(examples:$(OBJDIR)/%=%)

examples : $(examples)

$(ceed.pc) : pkgconfig-prefix = $(abspath .)
$(OBJDIR)/ceed.pc : pkgconfig-prefix = $(prefix)
.INTERMEDIATE: $(OBJDIR)/ceed.pc
%/ceed.pc : ceed.pc.template | $$(@D)/.DIR
	@sed "s:%prefix%:$(pkgconfig-prefix):" $< > $@

install : $(libceed) $(OBJDIR)/ceed.pc
	$(INSTALL) -d "$(DESTDIR)$(includedir)" "$(DESTDIR)$(libdir)" "$(DESTDIR)$(pkgconfigdir)"
	$(INSTALL_DATA) include/ceed.h "$(DESTDIR)$(includedir)/"
	$(INSTALL_DATA) $(libceed) "$(DESTDIR)$(libdir)/"
	$(INSTALL_DATA) $(OBJDIR)/ceed.pc "$(DESTDIR)$(pkgconfigdir)/"

.PHONY: all cln clean print test tst examples astyle install doc
cln clean :
	$(RM) *.o *.d $(libceed)
	$(RM) -r *.dSYM $(OBJDIR) $(LIBDIR)/pkgconfig
	$(MAKE) -C examples clean
	$(MAKE) -C examples/mfem clean
	cd examples/nek5000; bash make-nek-examples.sh clean; cd ../..;
	$(RM) $(magma.c) $(magma.cu) backends/magma/*~ backends/magma/*.o

distclean : clean
	rm -rf doc/html

doc :
	doxygen Doxyfile

astyle :
	astyle --style=google --indent=spaces=2 --max-code-length=80 \
            --keep-one-line-statements --keep-one-line-blocks --lineend=linux \
            --suffix=none --preserve-date --formatted \
            *.[ch] tests/*.[ch] backends/*/*.[ch] examples/*.[ch] examples/mfem/*.[ch]pp

print :
	@echo $(VAR)=$($(VAR))

print-%:
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

-include $(libceed.c:%.c=build/%.d) $(tests.c:tests/%.c=build/%.d)
