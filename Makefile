CFLAGS = -std=c99 -Wall -Wextra -Wno-unused-parameter -fPIC -lm
CFLAGS += $(if $(NDEBUG),-O2,-g)
CPPFLAGS = -I.

libfeme := libfeme.so
libfeme.c := $(wildcard feme*.c)
tests.c   := $(wildcard t[0-9][0-9]-*.c)
tests     := $(tests.c:%.c=%)

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
	$(RM) *.o $(libfeme) $(tests.c:%.c=%)

print :
	@echo $(VAR)=$($(VAR))
