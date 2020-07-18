# Directories
SRCDIR=src
OBJDIR=obj
EXTDIR=ext
SUBDIRS=$(dir $(wildcard $(SRCDIR)/*/.))

# Flags
CXX=g++
SUBFLAGS=$(addprefix -I, $(patsubst %/, %, $(SUBDIRS)))
CXXFLAGS=-g -Wall -O3 -fPIC -std=c++11 $(SUBFLAGS)
LDFLAGS=
LIBFLAGS=-pthread `pkg-config --libs opencv`

# Sources(/src) 
SRCS=$(wildcard $(SRCDIR)/*.cc)
HDRS=$(wildcard $(SRCDIR)/*.h) 
OBJS=$(SRCS:$(SRCDIR)/%.cc=$(OBJDIR)/%.o) 
EXESRCS=$(SRCDIR)/main.cc
EXEHDRS=
EXEOBJS=$(EXESRCS:$(SRCDIR)/%.cc=$(OBJDIR)/%.o)
# Sources(/src/*) 
SUBSRCS=$(wildcard $(SRCDIR)/*/*.cc)
SUBHDRS=$(wildcard $(SRCDIR)/*/*.h)
SUBOBJS=$(addprefix $(OBJDIR)/, $(notdir $(patsubst %.cc, %.o, $(SUBSRCS))))
# Executable
EXE=nebula
# Library
LIB=libnebula.so

# Targets
.PHONY: default 
default: $(LIB) $(EXE)

$(EXE): $(SUBOBJS) $(OBJS)
	@echo "# Makefile Target: $@" 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBFLAGS) 

$(LIB): $(SUBOBJS) $(filter-out $(EXEOBJS), $(OBJS))
	@echo "# Makefile Target: $@" 
	$(CXX) -g -shared -Wl,-soname,$@ $(SUBFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cc $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

$(OBJDIR)/%.o: $(SRCDIR)/*/%.cc $(SUBHDRS)
	@mkdir -pv $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

.PHONY: datasets
datasets:
	@./datasets/mklst.sh

.PHONY: clean
clean:
	@rm -f $(OBJS) $(SUBOBJS) $(EXE)
	@rm -rf $(OBJDIR)
	@echo "# Makefile Clean: $(OBJDIR)/'s [ $(notdir $(OBJS) $(SUBOBJS) ] and [ $(EXE)) ] are removed" 
