# Note: PETSC_ARCH can be undefined or empty for installations which do not use
#       PETSC_ARCH - for example when using PETSc installed through Spack.
PETSc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc
CEED_DIR ?= ../..
ceed.pc := $(CEED_DIR)/lib/pkgconfig/ceed.pc

CC = $(call pkgconf, --variable=ccompiler $(PETSc.pc) $(ceed.pc))
CFLAGS = -std=c99 $(call pkgconf, --variable=cflags_extra $(PETSc.pc)) $(call pkgconf, --cflags-only-other $(PETSc.pc)) $(OPT) -D__DIR__=\"$(dir $(abspath $(lastword $<)))\"
CPPFLAGS = $(call pkgconf, --cflags-only-I $(PETSc.pc) $(ceed.pc))
LDFLAGS = $(call pkgconf, --libs-only-L --libs-only-other $(PETSc.pc) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(call pkgconf, --variable=ldflag_rpath $(PETSc.pc))%, $(call pkgconf, --libs-only-L $(PETSc.pc) $(ceed.pc)))
LDLIBS = $(call pkgconf, --libs-only-l $(PETSc.pc) $(ceed.pc)) -lm

bpsraw.c := bpsraw.c
bpsraw := $(bpsraw.c:%.c=%)

bps.c := bps.c
bps := $(bps.c:%.c=%)

multigrid.c := multigrid.c
multigrid := $(multigrid.c:%.c=%)

area.c := area.c
area := $(area.c:%.c=%)

all: $(bpsraw) $(bps) $(multigrid) $(area)
$(bps): | $(PETSc.pc) $(ceed.pc)
$(bpsdmplex): | $(PETSc.pc) $(ceed.pc)
$(multigrid): | $(PETSc.pc) $(ceed.pc)
$(areadmplex): | $(PETSc.pc) $(ceed.pc)

# Rules for building the examples
%: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(abspath $<) -o $@ \
	  $(LDLIBS)

print: $(PETSc.pc) $(ceed.pc)
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))
	@true

clean:
	$(RM) $(bpsraw) $(bps) $(multigrid) $(area)

$(PETSc.pc):
	$(if $(wildcard $@),,$(error \
	  PETSc config not found at $@. Please set PETSC_DIR and PETSC_ARCH))

.PHONY: all print clean

pkgconf = $(shell pkg-config $1 | sed -e 's/^"//g' -e 's/"$$//g')
