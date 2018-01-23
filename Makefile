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

CC ?= gcc
FC ?= gfortran

NDEBUG ?=
LDFLAGS ?=
LOADLIBES ?=
TARGET_ARCH ?=
UNDERSCORE ?= 1

# env variable OCCA_DIR should point to OCCA-1.0 branch
OCCA_DIR ?= ../occa

SANTIZ = -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
CFLAGS = -std=c99 -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP -march=native
FFLAGS = -cpp     -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP -march=native

CFLAGS += $(if $(NDEBUG),-O2,-g)
ifeq ($(UNDERSCORE), 1)
  CFLAGS += -DUNDERSCORE
endif

FFLAGS += $(if $(NDEBUG),-O2,-g)

CFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
FFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
LDFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
CPPFLAGS = -I.
LDLIBS = -lm
OBJDIR := build
LIBDIR := .
NPROCS := $(shell getconf _NPROCESSORS_ONLN)
MFLAGS := -j $(NPROCS) --warn-undefined-variables \
			--no-print-directory --no-keep-going

PROVE ?= prove
PROVE_OPTS ?= -j $(NPROCS)
DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)
#libceed
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
# backends/[ref & occa]
ref.c     := $(sort $(wildcard backends/ref/*.c))
ref.o     := $(ref.c:%.c=$(OBJDIR)/%.o)
occa.c    := $(sort $(wildcard backends/occa/*.c))
occa.o    := $(occa.c:%.c=$(OBJDIR)/%.o)

# Output color rules
COLOR_OFFSET = 3
COLOR = $(shell echo $(rule_path)|cksum|cut -b1-2)
rule_path = $(notdir $(patsubst %/,%,$(dir $<)))
rule_file = $(basename $(notdir $@))
rule_dumb = @echo -e $(rule_path)/$(rule_file)
rule_term = @echo -e \\e[38\;5\;$(shell echo $(COLOR)+$(COLOR_OFFSET)|bc -l)\;1m\
             $(rule_path)\\033[m/\\033[\m$(rule_file)\\033[m
# if TERM=dumb, use it, otherwise switch to the term one
output = $(if $(TERM:dumb=),$(rule_term),$(rule_dumb))

V ?= 0
ifeq ($(V),0)
  quiet = @printf "  %10s %s\n" "$1" "$@"; $($(1))
else
  quiet = $($(1))
endif

.SUFFIXES:
.SUFFIXES: .c .o .d
.SECONDEXPANSION:		# to expand $$(@D)/.DIR

%/.DIR :
	@mkdir -p $(@D)
	@touch $@

.PRECIOUS: %/.DIR

all : $(libceed) ceed.pc

$(libceed) : LDFLAGS += $(if $(DARWIN), -install_name $(abspath $(libceed)))

$(libceed) : $(ref.o)
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  $(libceed) : LDFLAGS += -L$(OCCA_DIR)/lib -Wl,-rpath,$(abspath $(OCCA_DIR)/lib)
  $(libceed) : LDLIBS += -locca #-lrt -ldl
  $(libceed) : $(occa.o)
  $(occa.o) : CFLAGS += -I$(OCCA_DIR)/include
endif
$(libceed) : $(libceed.c:%.c=$(OBJDIR)/%.o)
	$(call quiet,CC) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : %.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/% : tests/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : tests/%.f | $$(@D)/.DIR
	$(call quiet,FC) $(FFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : examples/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(OBJDIR)/% : examples/%.f | $$(@D)/.DIR
	$(call quiet,FC) $(FFLAGS) $(LDFLAGS) -o $@ $(abspath $<) -lceed $(LDLIBS)

$(tests) $(examples) : $(libceed)
$(tests) $(examples) : LDFLAGS += -Wl,-rpath,$(abspath $(LIBDIR)) -L$(LIBDIR)
$(OBJDIR)/t% : tests/t%.c tests/t%.f $(libceed)

run-t% : $(OBJDIR)/t%
	@tests/tap.sh $(<:build/%=%)

test : $(tests:$(OBJDIR)/t%=run-t%)
tst:;@$(MAKE) $(MFLAGS) test

prove : $(tests)
	$(PROVE) $(PROVE_OPTS) --exec 'tests/tap.sh' $(tests:$(OBJDIR)/%=%)

examples : $(examples)

ceed.pc : ceed.pc.template
	@sed 's:%prefix%:$(abspath .):' $< > $@

.PHONY: all clean print test examples astyle
cln clean :
	$(RM) *.o $(OBJDIR)/*.o *.d $(OBJDIR)/*.d $(libceed) $(tests) ceed.pc
	$(RM) -r *.dSYM $(OBJDIR)/backends
	$(MAKE) -C examples/mfem clean

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
