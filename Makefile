# Directories
SRCDIR=src
OBJDIR=obj
EXTDIR=ext
SUBDIRS=$(dir $(wildcard $(SRCDIR)/*/.))

# Flags
CXX=g++
SUBFLAGS=$(addprefix -I, $(patsubst %/, %, $(SUBDIRS)))
CXXFLAGS=-g -Wall -O3 -std=c++11 $(SUBFLAGS)
LDFLAGS=
LIBFLAGS=-pthread `pkg-config --libs opencv`

# Sources(/src) 
SRCS=$(wildcard $(SRCDIR)/*.cc)
HDRS=$(wildcard $(SRCDIR)/*.h) 
OBJS=$(SRCS:$(SRCDIR)/%.cc=$(OBJDIR)/%.o) 
# Sources(/src/*) 
SUBSRCS=$(wildcard $(SRCDIR)/*/*.cc)
SUBHDRS=$(wildcard $(SRCDIR)/*/*.h)
SUBOBJS=$(addprefix $(OBJDIR)/, $(notdir $(patsubst %.cc, %.o, $(SUBSRCS))))
# Executable
EXE=nebula

# Targets
.PHONY: default directories
default: directories $(EXE)

$(EXE): $(SUBHDRS) $(HDRS) $(SUBOBJS) $(OBJS) 
	@echo "# Makefile Target: $@" 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.cc 
	@echo "# Makefile Target: $@" 
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

$(OBJDIR)/%.o: $(SRCDIR)/*/%.cc 
	@echo "# Makefile Target: $@" 
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

directories:
	@mkdir -p $(OBJDIR)

.PHONY: clean
clean:
	@rm -f $(OBJS) $(SUBOBJS) $(EXE)
	@rm -rf $(OBJDIR)
	@echo "# Makefile Clean: $(OBJDIR)/'s [ $(notdir $(OBJS) $(SUBOBJS) ] and [ $(EXE)) ] are removed" 
