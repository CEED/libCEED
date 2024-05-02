# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed
#
# The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

CONFIG ?= ../../config.mk
-include $(CONFIG)
COMMON ?= ../../common.mk
-include $(COMMON)

PETSc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc
CEED_DIR ?= ../..
ceed.pc := $(CEED_DIR)/lib/pkgconfig/ceed.pc

CC = $(call pkgconf, --variable=ccompiler $(PETSc.pc) $(ceed.pc))
CFLAGS = -std=c99 \
  $(call pkgconf, --variable=cflags_extra $(PETSc.pc)) \
  $(call pkgconf, --cflags-only-other $(PETSc.pc)) \
  $(OPT) $(OPT_EXAMPLES)
CPPFLAGS = $(call pkgconf, --cflags-only-I $(PETSc.pc) $(ceed.pc)) \
  $(call pkgconf, --variable=cflags_dep $(PETSc.pc))
CXX = $(call pkgconf, --variable=cxxcompiler $(PETSc.pc) $(ceed.pc))
CXXFLAGS = -std=c++17 -Wno-deprecated -Wno-tautological-compare
LDFLAGS = $(call pkgconf, --libs-only-L --libs-only-other $(PETSc.pc) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(PETSc.pc) $(ceed.pc)))
LDLIBS = $(call pkgconf, --libs-only-l $(PETSc.pc) $(ceed.pc)) -lm -lstdc++

# Address Sanitizer Setup
# ASAN must be left empty if you don't want to use it
ASAN ?=
AFLAGS ?= -fsanitize=address
# Also: -fsanitize=undefined -fno-omit-frame-pointer
CFLAGS += $(if $(ASAN),$(AFLAGS))
FFLAGS += $(if $(ASAN),$(AFLAGS))
LDFLAGS += $(if $(ASAN),$(AFLAGS))
CPPFLAGS += -I./include

# LibTorch
USE_LIBTORCH ?=

define TorchTryStatement
try:
	import torch.utils.cpp_extension
except ImportError:
	print("False")
	exit(0)
print("True")
endef

ifndef ($(USE_LIBTORCH))
  PYTHON_OUTPUT := $(shell python -c '$(TorchTryStatement)')
  ifeq ($(PYTHON_OUTPUT),True)
    USE_LIBTORCH := 1
  endif
endif

ifeq ($(USE_LIBTORCH),1)
  LIBTORCH_INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")
  LIBTORCH_LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.library_paths()[0])")
  LIBTORCH_USE_CUDA := $(shell python -c "import torch; print(torch.cuda.is_available())")
  ltorch := -ltorch -lc10 -ltorch_cpu

  CPPFLAGS += $(LIBTORCH_INCPATH) -D_GLIBCXX_USE_CXX11_ABI=1
  LDFLAGS += -L$(LIBTORCH_LIBPATH)
  LDFLAGS += -Wl,-rpath,$(LIBTORCH_LIBPATH)
  LDLIBS += $(ltorch)

  # Intel extension for pytorch (IPEX)
  IPEX_DIR ?=
  ifdef IPEX_DIR
      LDFLAGS += -L$(IPEX_DIR)/lib/
      LDFLAGS += -Wl,-rpath,$(IPEX_DIR)/lib/
      lipex = -lintel-ext-pt-gpu

      LDLIBS += $(lipex)
  endif
endif

# Source Files
OBJDIR := build
SRCDIR := src
PROBLEMDIR := problems

src.c := navierstokes.c $(sort $(wildcard $(PROBLEMDIR)/*.c)) $(sort $(wildcard $(SRCDIR)/*.c))
src.cpp := $(sort $(wildcard $(PROBLEMDIR)/*.cpp))
src.o = $(src.c:%.c=$(OBJDIR)/%.o) $(src.cpp:%.cpp=$(OBJDIR)/%.o)

# Path to install directory for SmartRedis. Example: /software/smartredis/install
SMARTREDIS_DIR ?=
ifdef SMARTREDIS_DIR
	hiredis.pc := $(SMARTREDIS_DIR)/lib/pkgconfig/hiredis.pc
	lsmartredis:= -lsmartredis
	redis++.pc = $(wildcard $(SMARTREDIS_DIR)/lib/pkgconfig/redis++.pc $(SMARTREDIS_DIR)/lib64/pkgconfig/redis++.pc)

	CPPFLAGS += $(call pkgconf, --cflags-only-I $(hiredis.pc) $(redis++.pc))
	LDFLAGS += $(call pkgconf, --libs-only-L --libs-only-other $(hiredis.pc) $(redis++.pc))
	LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(hiredis.pc) $(redis++.pc)))
	LDLIBS += $(call pkgconf, --libs-only-l $(hiredis.pc) $(redis++.pc)) $(lsmartredis)
	src.c += $(sort $(wildcard $(SRCDIR)/smartsim/*.c))
endif

all: navierstokes

navierstokes: $(src.o) | $(PETSc.pc) $(ceed.pc)
	$(call quiet,LINK.o) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

.SECONDEXPANSION: # to expand $$(@D)/.DIR
%/.DIR :
	@mkdir -p $(@D)
	@touch $@

# Quiet, color output
quiet ?= $($(1))

$(OBJDIR)/%.o : %.c | $$(@D)/.DIR
	$(call quiet,CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(abspath $<)

$(OBJDIR)/%.o : %.cpp | $$(@D)/.DIR
	$(call quiet,CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $(abspath $<)

print: $(PETSc.pc) $(ceed.pc)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	$(info OPT     : $(OPT))
	@true

print-% :
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [        flavor]: $(flavor $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true

clean:
	$(RM) -r $(OBJDIR) navierstokes *.vtu *.bin* *.csv *.png

$(PETSc.pc):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all print clean

pkgconf = $(shell pkg-config $(if $(STATIC),--static) $1 | sed -e 's/^"//g' -e 's/"$$//g')

-include $(src.o:%.o=%.d)
