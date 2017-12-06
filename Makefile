CFLAGS = -std=c99 -Wall -Wextra -Wno-unused-parameter -fPIC -MMD -MP
CFLAGS += $(if $(NDEBUG),-O2,-g)
CPPFLAGS = -I.
LDLIBS = -lm

DARWIN := $(filter Darwin,$(shell uname -s))
SO_EXT := $(if $(DARWIN),dylib,so)

libfeme := libfeme.$(SO_EXT)
libfeme.c := $(wildcard feme*.c)
tests.c   := $(sort $(wildcard t[0-9][0-9]-*.c))
tests     := $(tests.c:%.c=%)

.SUFFIXES:
.SUFFIXES: .c .o .d

$(libfeme) : $(libfeme.c:%.c=%.o)
	$(CC) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

$(tests) : $(libfeme)
$(tests) : LDFLAGS += -Wl,-rpath,. -L.
t% : t%.c $(libfeme)

run-t% : t%
	@./tap.sh $<

test : $(tests:%=run-%)

.PHONY: clean print
clean :
	$(RM) *.o *.d $(libfeme) $(tests.c:%.c=%)
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

-include $(libfeme.c:%.c=%.d) $(tests.c:%.c=%.d)
