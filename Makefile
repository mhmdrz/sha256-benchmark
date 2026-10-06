# SHA-256 arm64 benchmark

CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra -std=c11
LDFLAGS ?=

UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

OBJ := sha256.o sha256_scalar.o sha256_fast.o bench.o

ifneq (,$(filter arm64 aarch64,$(UNAME_M)))
  CPPFLAGS += -DHAVE_ARMV8_SHA2
  OBJ      += sha256_armv8.o sha256_mb.o
  ifeq ($(UNAME_S),Darwin)
    ARMV8_CFLAGS :=
  else
    ARMV8_CFLAGS := -march=armv8-a+crypto
  endif
endif

bench: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ)

sha256_armv8.o: sha256_armv8.c sha256.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(ARMV8_CFLAGS) -c -o $@ $<

sha256_mb.o: sha256_mb.c sha256.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(ARMV8_CFLAGS) -c -o $@ $<

%.o: %.c sha256.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

run: bench
	./bench

clean:
	rm -f bench $(OBJ)

.PHONY: run clean
