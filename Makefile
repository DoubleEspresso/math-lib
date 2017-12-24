# architecture
ARCH = $(shell uname -p)
BITS = $(shell uname -m)
OS = $(shell uname) 
OSFLAVOR =

USERMACROS = -DNDEBUG -DTHREADED
INC = . /usr/local/cuda/include
CFLAGS = $(foreach id,$(INC),-I$(id))
CFLAGS += -Wall -O3 -fomit-frame-pointer -fstrict-aliasing -ffast-math -std=gnu++11 -pthread
CUFLAGS = -std=c++11 -c -arch=sm_20
DFLAGS =
LFLAGS = -lpthread -lrt -L/usr/local/cuda/lib64/stubs -lcuda -L/usr/local/cuda/lib64 -lcudart

#CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
#                -gencode arch=compute_21,code=sm_21 \
#                -gencode arch=compute_30,code=sm_30 \
#                -gencode arch=compute_35,code=sm_35 \
#                -gencode arch=compute_50,code=sm_50 \
#                -gencode arch=compute_52,code=sm_52 \
#		-gencode arch=compute_60,code=sm_60

CUDA_ARCH :=	-gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60

NVFLAGS := -O3 -rdc=true #rdc needed for separable compilation
NVFLAGS += $(CUDA_ARCH)
NVFLAGS += $(foreach id,$(INC),-I$(id))


INSTALL = /usr/local/bin

EXE_BITS =
EXE_OS =
THREADED =
GIT_VERSION =

# build info
SRC := main.cpp vegas.cpp
CU_SRC := matrix.cu matrix_lu.cu vector.cu

# gpu compiler
NVCC = nvcc

# cpu compiler
ifeq ($(COMP),)
   CC = g++
else
   CC = $(COMP)
endif

# arch
ifeq ($(ARCH),i386)
   CFLAGS += -sse
   CFLAGS += -O2
endif

# bits
ifeq ($(BITS),x86_64)
   EXE_BITS = 64
   CFLAGS += -m64
   USERMACROS += -DBIT_64
else
   EXE_BITS = 32
   CFLAGS += -m32
   USERMACROS += -DBIT_32
endif

# prefetch
ifeq ($(USE_PREFETCH),true)
   USERMACROS += -DUSE_PREFETCH
endif

# os
ifeq ($(OS),Darwin )
   EXE_OS = osx
   USERMACROS += -DOS=\"unix\"
endif

ifeq ($(OS),Linux )
   EXE_OS = nix
   USERMACROS += -DOS=\"unix\"
endif

# executable
EXE = mlib.exe

# git version info
GIT_VERSION := $(shell git describe --abbrev=4 --dirty --always --tags)
USERMACROS += -DBUILD_DATE="\"$$(date)\""
USERMACROS += -DVERSION=\"$(GIT_VERSION)\"

OBJ := $(patsubst %.cu, %.cu.o, $(filter %.cu,$(CU_SRC)))
#OBJ += $(patsubst %.cpp, %.o, $(filter %.cpp,$(SRC)))	



.PHONY: all
.SUFFIXES: .cpp .cu .o

all: information link

debug: CFLAGS += -g -ggdb
debug: NVFLAGS = -O0 $(CUDA_ARCH) -g -G
debug: NVFLAGS += $(foreach id,$(INC),-I$(INC))
debug: USERMACROS:=$(filter-out -DNDEBUG, $(USERMACROS))
debug: USERMACROS += -DDEBUG
debug: all

information:
	@echo ".............................."
	@echo "...ARCH    = "$(ARCH)
	@echo "...BITS    = "$(BITS)
	@echo "...CC      = "$(CC)
	@echo "...OS      = "$(OS)
	@echo "...CFLAGS  = "$(CFLAGS)
	@echo "...MACROS  = "$(USERMACROS)
	@echo "...EXE     = "$(EXE)
	@echo "..............................."
	@echo ""

%.cu.o:%.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@

#%.o:%.cpp
#	$(CC) -fPIC $(CFLAGS) $(USERMACROS) $< -o $@

link: $(OBJ)
	$(NVCC) -o $(EXE) $(SRC) $(OBJ) $(LFLAGS)

install: all
	if [ ! -d $(INSTALL) ]; then \
		mkdir -p $(INSTALL); \
	fi 
	mv $(EXE) $(INSTALL)
	find . -name "*.o" | xargs rm -vf 

TAGS:   $(SRC)
	etags $(SRC)

depend:
	makedepend -- $(DFLAGS) -- $(SRC)

.PHONY:clean
clean:
	find . -name "*.o" | xargs rm -vf
	find . -name "*.ii" | xargs rm -vf
	find . -name "*.s" | xargs rm -vf
	rm -vf *~ *#
