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

NDEBUG ?=
LDFLAGS ?= 
LOADLIBES ?=
TARGET_ARCH ?=

# env variable OCCA_DIR should point to OCCA-1.0 branch
OCCA_DIR ?= ../occa

pwd = $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

SANTIZ = -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
CFLAGS = -std=c99 -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP -march=native
CFLAGS += $(if $(NDEBUG),-O2,-g)
CFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
LDFLAGS += $(if $(NDEBUG),,)#$(SANTIZ))
CPPFLAGS = -I.
LDLIBS = -lm
OBJDIR := build
LIBDIR := .
NPROCS := $(shell getconf _NPROCESSORS_ONLN)
MFLAGS := -j $(NPROCS) --warn-undefined-variables \
			--no-print-directory --no-keep-going

PROVE ?= prove
DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)

libceed := $(LIBDIR)/libceed.$(SO_EXT)
libceed.c := $(wildcard ceed*.c)
tests.c   := $(sort $(wildcard tests/t[0-9][0-9]-*.c))
tests     := $(tests.c:tests/%.c=$(OBJDIR)/%)
examples.c := $(sort $(wildcard examples/*.c))
examples  := $(examples.c:examples/%.c=$(OBJDIR)/%)
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

all dbg:; @$(MAKE) $(MFLAGS) $(libceed) $(tests)
opt:;NDEBUG=1 $(MAKE) $(MFLAGS) $(libceed) $(tests)

$(libceed) : $(ref.o)
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
  $(libceed) : LDFLAGS += -L$(OCCA_DIR)/lib -Wl,-rpath,$(OCCA_DIR)/lib
  $(libceed) : LDLIBS += -locca -lrt -ldl
  $(libceed) : $(occa.o)
  $(occa.o) : CFLAGS += -I$(OCCA_DIR)/include
endif
$(libceed) : $(libceed.c:%.c=$(OBJDIR)/%.o) $(ref.c:%.c=$(OBJDIR)/%.o)
	$(call quiet,CC) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : $(pwd)/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/% : $(pwd)/tests/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : $(pwd)/examples/%.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(tests) $(examples) : $(libceed)
$(tests) $(examples) : LDFLAGS += -Wl,-rpath,$(LIBDIR) -L$(LIBDIR)
$(OBJDIR)/t% : tests/t%.c $(libceed)
$(OBJDIR)/ex% : examples/ex%.c $(libceed)

run-t% : $(OBJDIR)/t%
	tests/tap.sh $(<:build/%=%)

test : $(tests:$(OBJDIR)/t%=run-t%)
tst:;@$(MAKE) $(MFLAGS) test

prove : $(tests)
	$(PROVE) -j $(NPROCS) --exec 'tests/tap.sh' $(tests:$(OBJDIR)/%=%)

examples : $(examples)

.PHONY: clean print test examples astyle
cln clean :
	$(RM) *.o $(OBJDIR)/*.o *.d $(OBJDIR)/*.d $(libceed) $(tests)
	$(RM) -r *.dSYM $(OBJDIR)/backends

### ASTYLE ###
ASTYLE = astyle --options=.astylerc
FORMAT_FILES = $(foreach dir,. tests examples backends/ref backends/occa,$(dir)/*.[ch])
style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi

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
