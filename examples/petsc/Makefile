PETSc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc
ceed.pc := ../../ceed.pc

CC := $(shell pkg-config --variable=ccompiler $(PETSc.pc) $(ceed.pc))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PETSc.pc)) $(shell pkg-config --cflags-only-other $(PETSc.pc))
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PETSc.pc) $(ceed.pc))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PETSc.pc) $(ceed.pc))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PETSc.pc))%, $(shell pkg-config --libs-only-L $(PETSc.pc) $(ceed.pc)))
LDLIBS := $(shell pkg-config --libs-only-l $(PETSc.pc) $(ceed.pc)) -lm

bp.c := $(wildcard bp*.c)
bp := $(bp.c:%.c=%)

all: $(bp)

print:
	$(info CC      : $(CC))
	$(info CFLAGS  : $(CFLAGS))
	$(info CPPFLAGS: $(CPPFLAGS))
	$(info LDFLAGS : $(LDFLAGS))
	$(info LDLIBS  : $(LDLIBS))

clean:
	$(RM) $(bp)

.PHONY: all print clean
