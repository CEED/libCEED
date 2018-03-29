PETSc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc
CEED_DIR ?= ../..
ceed.pc := $(CEED_DIR)/lib/pkgconfig/ceed.pc

CC = $(call pkgconf, --variable=ccompiler $(PETSc.pc) $(ceed.pc))
CFLAGS = -std=c99 $(call pkgconf, --variable=cflags_extra $(PETSc.pc)) $(call pkgconf, --cflags-only-other $(PETSc.pc))
CPPFLAGS = $(call pkgconf, --cflags-only-I $(PETSc.pc) $(ceed.pc))
LDFLAGS = $(call pkgconf, --libs-only-L --libs-only-other $(PETSc.pc) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(PETSc.pc) $(ceed.pc)))
LDLIBS = $(call pkgconf, --libs-only-l $(PETSc.pc) $(ceed.pc)) -lm

bp.c := $(wildcard bp*.c)
bp := $(bp.c:%.c=%)

all: $(bp) okl-lib-cache
$(bp): | $(PETSc.pc) $(ceed.pc)

print: $(PETSc.pc) $(ceed.pc)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	@true

# okl-lib-cache depending on OCCA and INSTALL
okl-lib-cache:
# First test to see if libceed was compiled with OCCA
ifneq ($(wildcard $(OCCA_DIR)/lib/libocca.*),)
# Second test to see if '../..' is CEED_DIR
ifneq ($(CEED_DIR),../..)
okl-lib-cache:
	$(OCCA_DIR)/bin/occa cache ceed $(abspath $(CEED_DIR)/lib/okl/*.okl)
endif
endif

clean:
	$(RM) $(bp)

$(PETSc.pc):
	$(error PETSc config not found. Please set PETSC_DIR and PETSC_ARCH)

.PHONY: all print clean

pkgconf = $(shell pkg-config $1 | sed -e 's/^"//g' -e 's/"$$//g')
