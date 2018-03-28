PETSc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc
CEED_DIR ?= ../..
ceed.pc := $(CEED_DIR)/lib/pkgconfig/ceed.pc

CC := $(shell pkg-config --variable=ccompiler $(PETSc.pc) $(ceed.pc))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PETSc.pc)) $(shell pkg-config --cflags-only-other $(PETSc.pc))
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PETSc.pc) $(ceed.pc))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PETSc.pc) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PETSc.pc))%, $(shell pkg-config --libs-only-L $(PETSc.pc) $(ceed.pc)))
LDLIBS := $(shell pkg-config --libs-only-l $(PETSc.pc) $(ceed.pc)) -lm

bp.c := $(wildcard bp*.c)
bp := $(bp.c:%.c=%)

all: $(bp) okl-lib-cache

print:
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))

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

.PHONY: all print clean
