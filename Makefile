SRC ?= main.c

SRCDIR ?= .
OBJDIR ?= .objdir
DEPDIR ?= .depdir
LIBDIR := libs
LIB := $(notdir $(wildcard $(LIBDIR)/*))

##########
# less important stuff
SHELL := /bin/sh
MKDIR ?= mkdir
RMDIR ?= rmdir

COLOR := \033[1;34m
RESET := \033[0m

##########
# expand variables now
SRC := $(SRC)

MKDIR := $(MKDIR)
RMDIR := $(RMDIR)

##########
# warnings

# enable a lot of warnings and then some more
WARNINGS := -Wall -Wextra
# shadowing variables, are you sure?
WARNINGS += -Wshadow
# sizeof(void)
WARNINGS += -Wpointer-arith
# unsafe pointer cast qualifiers: `const char*` is cast to `char*`
WARNINGS += -Wcast-qual
# most of the time you don't want this
WARNINGS += -Werror=implicit-function-declaration
# let's stick with C89
# WARNINGS += -Werror=declaration-after-statement
# why warning for comments inside comments
WARNINGS += -Wno-comment

# be more strict
ifeq ($(shell $(CC) -dumpversion),4.7)
	# welcome to 2012
	# -Wpedantic is available since gcc 4.8
	WARNINGS += -pedantic
else
	WARNINGS += -Wpedantic
endif

# functions should be declared
#WARNINGS += -Wmissing-declarations

##########

# C-specific warnings
CWARNINGS := $(WARNINGS)
# warn about fn() instead of fn(void)
CWARNINGS += -Werror=strict-prototypes
# this probably should be enabled only in bigger projects
#CWARNINGS += -Wmissing-prototypes

MARCH := -mmmx -msse -msse2 -msse3 -mssse3 -mlong-double-64 -mno-fxsr -mno-cx16
MARCH += -mtune=core2

# standards (ANSI C, ANSI C++)
CFLAGS ?= $(CWARNINGS) -std=c11 -O2 -fstack-protector-strong -m64 $(MARCH)
CXXFLAGS ?= $(WARNINGS) -std=c++14 -O2 -fstack-protector-strong -m64 $(MARCH)
# for future use if needed
DEPFLAGS ?= -MP
LDLIBS += -lm
# if there are any cpp files, link with c++ library
ifneq ($(filter %.cpp,$(SRC)),)
	LDLIBS += -lstdc++
endif

# static link
LDFLAGS += -static -static-libgcc
# remove all symbol table and relocation information from the executable
LDFLAGS += -s

##########

# Intel MKL
CFLAGS += -isystem"/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/include"
CXXFLAGS += -isystem"/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/include"
LDLIBS += -Wl,--start-group
LDLIBS += -lmkl_intel_ilp64
ifeq ($(thread),gnu)
	LDLIBS += -lmkl_gnu_thread
else
	LDLIBS += -lmkl_intel_thread
endif
LDLIBS += -lmkl_core
LDLIBS += -Wl,--end-group

ifeq ($(thread),gnu)
	LDLIBS += -lgomp
else
	LDLIBS += -liomp5
endif
LDLIBS += -lpthread
LDLIBS += -lm
LDLIBS += -ldl
LDFLAGS += -L"/opt/intel/compilers_and_libraries_2016.2.181/linux/mkl/lib/intel64"
LDFLAGS += -L"/opt/intel/compilers_and_libraries_2016.2.181/linux/compiler/lib/intel64"

# add unicode support
CFLAGS += $(shell pkg-config --cflags-only-I icu-uc icu-io)
CXXFLAGS += $(shell pkg-config --cflags-only-I icu-uc icu-io)
LDLIBS += $(shell pkg-config --libs-only-l icu-uc icu-io)
LDFLAGS += $(shell pkg-config --libs-only-L icu-uc icu-io)

EXE := $(basename $(firstword $(SRC)))

OBJ := $(foreach src,$(SRC),$(OBJDIR)/$(src).o)

DEP := $(foreach src,$(SRC),$(DEPDIR)/$(src).d)

LIBDEP := $(foreach lib,$(LIB),$(LIBDIR)/$(lib)/lib$(lib).a)

# add libraries
LDFLAGS += $(foreach lib,$(LIB),-L$(LIBDIR)/$(lib)/)
CFLAGS += $(foreach lib,$(LIB),-I$(LIBDIR)/$(lib)/)
CXXFLAGS += $(foreach lib,$(LIB),-I$(LIBDIR)/$(lib)/)

STYLE := $(filter %.c %.h %.cpp %.hpp,$(SRC))

STYLED := $(foreach src,$(STYLE),$(DEPDIR)/$(src).styled)

# be silent unless VERBOSE
ifndef VERBOSE
.SILENT: ;
endif

# template for inline assembler
define INCBIN :=
  .section @file@, "a"
  .global @sym@_start
@sym@_start:
  .incbin "@file@"
  .global @sym@_end
@sym@_end:

endef
export INCBIN

# default target
.PHONY: all
all: $(EXE) ## build executable

.PHONY: run
run: $(EXE) ## run program
	@./$(EXE)

.PHONY: debug
debug: CFLAGS := $(filter-out -O2,$(CFLAGS)) -D_DEBUG -g
debug: CXXFLAGS := $(filter-out -O2,$(CXXFLAGS)) -D_DEBUG -g
debug: LDFLAGS := $(filter-out -s,$(LDFLAGS))
debug: $(EXE) ## build with debug enabled

.PHONY: debugrun
debugrun: debug run ## run debug version

.PHONY: style
style: $(STYLED)

$(DEPDIR)/%.styled: $(SRCDIR)/%
	@printf "$(COLOR)Styling $(SRCDIR)/$*$(RESET)\\n"
# sed is needed to fix string literals (uncrustify bug: https://github.com/uncrustify/uncrustify/issues/945)
	uncrustify -c .uncrustify.cfg --replace --no-backup $(SRCDIR)/$* && sed -i -e 's/\([uUL]\)\s\+\(['"'"'"]\)/\1\2/g' $(SRCDIR)/$* && touch $@

# link
$(EXE): $(OBJ) $(LIBDEP)
	@printf "$(COLOR)Link $^ -> $@$(RESET)\\n"
	$(CC) -Wl,--as-needed -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS)

# create binary blobs for other files
$(filter-out %.c.o %.cpp.o,$(OBJ)): $(OBJDIR)/%.o: $(SRCDIR)/%
	@printf "$(COLOR)Other file $(SRCDIR)/$* -> $@$(RESET)\\n"
	echo "$$INCBIN" | sed -e 's/@sym@/$(subst .,_,$*)/' -e 's/@file@/$</' | gcc -x assembler-with-cpp - -c -o $@

# compile
$(OBJDIR)/%.c.o: $(DEPDIR)/%.c.d
	@printf "$(COLOR)Compile $(SRCDIR)/$*.c -> $@$(RESET)\\n"
	$(CC) $(CFLAGS) -c -o $@ $(SRCDIR)/$*.c

# compile
$(OBJDIR)/%.cpp.o: $(DEPDIR)/%.cpp.d
	@printf "$(COLOR)Compile $(SRCDIR)/$*.cpp -> $@$(RESET)\\n"
	$(CXX) $(CXXFLAGS) -c -o $@ $(SRCDIR)/$*.cpp

# build dependecies list
$(DEPDIR)/%.c.d: $(SRCDIR)/%.c
	@printf "$(COLOR)Generating dependencies $(SRCDIR)/$*.c -> $@$(RESET)\\n"
	$(CC) $(CFLAGS) $(DEPFLAGS) -MM -MT '$$(OBJDIR)/$*.c.o' $< | sed 's,^\([^:]\+.o\):,\1 $$(DEPDIR)/$*.c.d:,' > $@

# build dependecies list
$(DEPDIR)/%.cpp.d: $(SRCDIR)/%.cpp
	@printf "$(COLOR)Generating dependencies $(SRCDIR)/$*.cpp -> $@$(RESET)\\n"
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -MM -MT '$$(OBJDIR)/$*.cpp.o' $< | sed 's,^\([^:]\+.o\):,\1 $$(DEPDIR)/$*.cpp.d:,' > $@

# include generated dependencies
-include $(wildcard $(DEP))

# depend on directory
$(OBJ): | $(OBJDIR)
$(DEP): | $(DEPDIR)
$(STYLED): | $(DEPDIR)

# create directory
$(OBJDIR):
	-$(MKDIR) $(OBJDIR)

# create directory
$(DEPDIR):
	-$(MKDIR) $(DEPDIR)

# make libraries
$(LIBDEP): %: force_check
	@printf "$(COLOR)Verifying $@$(RESET)\\n"
	$(MAKE) CFLAGS="$(filter-out -W%,$(CFLAGS))" -C $(@D) $(@F)

.PHONY: force_check

# delete stuff
.PHONY: clean
clean: mostlyclean ## delete everything this Makefile created
	-$(RM) $(EXE)

.PHONY: mostlyclean
mostlyclean: ## delete everything created, leave executable
	@printf "$(COLOR)Cleaning$(RESET)\\n"
ifneq ($(wildcard $(OBJDIR)),)
	-$(RM) $(OBJ)
	-$(RMDIR) $(OBJDIR)
endif
ifneq ($(wildcard $(DEPDIR)),)
	-$(RM) $(DEP)
	-$(RM) $(STYLED)
	-$(RMDIR) $(DEPDIR)
endif

.PHONY: forceclean
forceclean: ## force delete all created temporary folders
	@printf "$(COLOR)Force cleaning$(RESET)\\n"
ifneq ($(wildcard $(OBJDIR)),)
	-$(RM) -r $(OBJDIR)
endif
ifneq ($(wildcard $(DEPDIR)),)
	-$(RM) -r $(DEPDIR)
endif

.PHONY: help
help: ## show this help
	@awk -F':.*##' '/: [^#]*##/{ printf("%12s: %s\n", $$1, $$2)}' $(MAKEFILE_LIST)

