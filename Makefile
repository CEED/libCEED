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

CFLAGS = -std=c99 -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP
CFLAGS += $(if $(NDEBUG),-O2,-g)
CPPFLAGS = -I.
LDLIBS = -lm
OBJDIR := build
LIBDIR := .
NPROCS := $(shell getconf _NPROCESSORS_ONLN)

PROVE ?= prove
DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)

libceed := $(LIBDIR)/libceed.$(SO_EXT)
libceed.c := $(wildcard ceed*.c)
tests.c   := $(sort $(wildcard tests/t[0-9][0-9]-*.c))
tests     := $(tests.c:tests/%.c=$(OBJDIR)/%)
examples.c := $(sort $(wildcard examples/*.c))
examples  := $(examples.c:examples/%.c=$(OBJDIR)/%)

.SUFFIXES:
.SUFFIXES: .c .o .d
.SECONDEXPANSION:		# to expand $$(@D)/.DIR

%/.DIR :
	@mkdir -p $(@D)
	@touch $@

.PRECIOUS: %/.DIR

all:;$(MAKE) --no-print-directory -j $(NPROCS) $(libceed)

$(libceed) : $(libceed.c:%.c=$(OBJDIR)/%.o)
	$(CC) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : %.c | $$(@D)/.DIR
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $^

$(OBJDIR)/%.o : tests/%.c | $$(@D)/.DIR
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $^

$(OBJDIR)/%.o : examples/%.c | $$(@D)/.DIR
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $^

$(tests) $(examples) : $(libceed)
$(tests) $(examples) : LDFLAGS += -Wl,-rpath,$(LIBDIR) -L$(LIBDIR)
$(OBJDIR)/t% : tests/t%.c $(libceed)
$(OBJDIR)/ex% : examples/ex%.c $(libceed)

run-t% : $(OBJDIR)/t%
	@tests/tap.sh $(<:build/%=%)

test : $(tests:$(OBJDIR)/t%=run-t%)
tst:;@$(MAKE) --no-print-directory -j $(NPROCS) test

prove : $(tests)
	$(PROVE) --exec tests/tap.sh $(CEED_PROVE_OPTS) $(tests:$(OBJDIR)/%=%)

examples : $(examples)

.PHONY: clean print test examples astyle
cln clean :
	$(RM) *.o $(OBJDIR)/*.o *.d $(OBJDIR)/*.d $(libceed) $(tests.c:%.c=%)
	$(RM) -r *.dSYM

astyle :
	astyle --style=google --indent=spaces=2 --max-code-length=80 \
            --keep-one-line-statements --keep-one-line-blocks --lineend=linux \
            --suffix=none --preserve-date --formatted \
            *.[ch] tests/*.[ch] examples/*.[ch]


print :
	@echo $(VAR)=$($(VAR))

print-%:
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

-include $(libceed.c:%.c=%.d) $(tests.c:%.c=%.d)
