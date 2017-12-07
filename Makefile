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

DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)

libceed := libceed.$(SO_EXT)
libceed.c := $(wildcard ceed*.c)
tests.c   := $(sort $(wildcard t/t[0-9][0-9]-*.c))
tests     := $(tests.c:%.c=%)

.SUFFIXES:
.SUFFIXES: .c .o .d

$(libceed) : $(libceed.c:%.c=%.o)
	$(CC) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(tests) : $(libceed)
$(tests) : LDFLAGS += -Wl,-rpath,. -L.
t/t% : t/t%.c $(libceed)

run-t% : t/t%
	@./tap.sh $(<:t/%=%)

test : $(tests:t/%=run-%)

.PHONY: clean print
clean :
	$(RM) *.o t/*.o *.d t/*.d $(libceed) $(tests.c:%.c=%)
	$(RM) -r *.dSYM

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
