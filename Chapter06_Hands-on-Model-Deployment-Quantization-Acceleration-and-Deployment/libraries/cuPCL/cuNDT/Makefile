################################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

#Get default CUDA version installed by dep package
CUDAVERSION ?= cuda-$(shell dpkg -l | grep cuda-core | sed -e "s/ \{1,\}/ /g" | cut -d ' ' -f 3 | cut -d '.' -f 1,2 | sed -e "s/-.*//g" | sort -n | tail -n 1)

CHECK_CUDA := 0
CHECK_CUDA := $(shell if [ -x "/usr/local/$(CUDAVERSION)" ]; then echo 1; fi;)

CUDNN_PATH ?=

ifneq ($(CHECK_CUDA), 1)
    #no version info, use cuda default path
    CUDAVERSION := cuda
    CHECK_CUDA := $(shell if [ -x "/usr/local/$(CUDAVERSION)" ]; then echo 1; fi;)
    $(info USE Default CUDA DIR: /usr/local/$(CUDAVERSION))
    ifneq ($(CHECK_CUDA), 1)
        $(error $("Please install cuda package"))
    endif
endif

LIBDIR := lib64

TARGET_ARCH ?= $(shell uname -m)

$(info TARGET_ARCH: $(TARGET_ARCH))

ifeq ($(TARGET_ARCH), aarch64)
    ifeq ($(shell uname -m), aarch64)
        CC = g++
    else
        CC = aarch64-linux-gnu-g++
    endif
    NVCC = /usr/local/$(CUDAVERSION)/bin/nvcc -m64 -ccbin $(CC)
else ifeq ($(TARGET_ARCH), x86_64)
    CC = g++
    NVCC = /usr/local/$(CUDAVERSION)/bin/nvcc -m64
else
    $(error Auto-detection of platform failed. Please specify one of the following arguments to make: TARGET_ARCH=[aarch64|x86_64])
endif

CXXFLAGS        += -std=c++14 -O2
CCFLAGS         += -D_REENTRANT
LDFLAGS         += -Wl,--allow-shlib-undefined -pthread
#CCFLAGS         += -D_GLIBCXX_USE_CXX11_ABI=0

dbg ?= 0
# show libraries used by linker in debug mode
ifeq ($(dbg),1)
    $(info dbg: $(dbg))
	CCFLAGS     += -g
	NVCCFLAGS   += -G --ptxas-options=-v
	LDFLAGS += -Wl,--trace
endif

ifeq ($(TARGET_ARCH), x86_64)
CUDA_VERSION := $(shell cat /usr/local/$(CUDAVERSION)/targets/x86_64-linux/include/cuda.h |grep "define CUDA_VERSION" |awk '{print $$3}') 
endif
ifeq ($(TARGET_ARCH), ppc64le)
CUDA_VERSION := $(shell cat /usr/local/$(CUDAVERSION)/targets/ppc64le-linux/include/cuda.h |grep "define CUDA_VERSION" |awk '{print $$3}') 
endif
ifeq ($(TARGET_ARCH), aarch64)
CUDA_VERSION := $(shell cat /usr/local/$(CUDAVERSION)/targets/aarch64-linux/include/cuda.h |grep "define CUDA_VERSION" |awk '{print $$3}') 
endif

CUDA_VERSION := $(strip $(CUDA_VERSION))
$(info CUDA_VERSION: $(CUDA_VERSION))

ifeq ($(CUDA_VERSION),8000)
  SMS_VOLTA = 
else
  ifneq ($(TARGET_ARCH),ppc64le)
    ifeq ($(CUDA_VERSION),9000)
      SMS_VOLTA ?= 70 
    else
      SMS_VOLTA ?= 70 72
    endif
  else
    SMS_VOLTA ?= 70 
  endif
endif

ifeq ($(TARGET_ARCH), aarch64)
    ifeq ($(CUDA_VERSION), 9000)
      SMS_VOLTA := 62 70
    endif
endif

ifeq ($(CUDA_VERSION),10020)
SMS_TURING ?= 75
endif

ifeq ($(CUDA_VERSION),11040)
SMS_AMPERE ?= 87
endif

# Gencode arguments
SMS ?= 30 35 37 50 52 53 60 61 62 $(SMS_VOLTA) $(SMS_TURING) $(SMS_AMPERE)
$(info SMS: $(SMS))

ifeq ($(GENCODE_FLAGS),)
    # Generate SASS code for each SM architecture listed in $(SMS)
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

    ifeq ($(SMS),)
        # Generate PTX code from SM 20
        GENCODE_FLAGS += -gencode arch=compute_53,code=sm_53
    endif
    # Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
    HIGHEST_SM := $(lastword $(sort $(SMS)))
    ifneq ($(HIGHEST_SM),)
        GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
    endif
endif

CUDA_CFLAGS := -I/usr/local/$(CUDAVERSION)/include
CUDA_LIBS   := -L/usr/local/$(CUDAVERSION)/$(LIBDIR) -lcudart_static -lrt -ldl -lpthread -lcudart

CUDA_CFLAGS += -I$(CUDNN_PATH)/include
CUDA_LIBS   += -L$(CUDNN_PATH)/lib64 -lcudnn

INCLUDE     :=
INCLUDE     += $(CUDA_CFLAGS)
INCLUDE     += -I/usr/local/include
INCLUDE     += -I/usr/include/eigen3/ -I/usr/include/pcl-1.10/ -I/usr/include/vtk-6.3/

LIBRARIES   :=
LIBRARIES   += -L/usr/lib
LIBRARIES   += -L/usr/local/lib
LIBRARIES   += $(CUDA_LIBS)
LIBRARIES   += -lpthread
LIBRARIES   += -L/usr/lib/aarch64-linux-gnu/ -lboost_system -lpcl_common -lpcl_io -lpcl_recognition -lpcl_features -lpcl_sample_consensus -lpcl_octree -lpcl_search -lpcl_filters -lpcl_kdtree -lpcl_segmentation -lpcl_visualization

OBJ_DIR     := obj

CPP_FILES       := $(wildcard *.cpp)
CU_FILES        := $(wildcard *.cu)
LIBRARY_FILES   := $(wildcard ./lib/*.so)

OBJ_FILES_CPP    := $(CPP_FILES:%.cpp=$(OBJ_DIR)/%.o)
OBJ_FILES_CU    := $(CU_FILES:%.cu=$(OBJ_DIR)/%.o)

TARGET         := demo

all: $(OBJ_DIR) $(TARGET)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(OBJ_FILES_CPP): $(OBJ_DIR)/%.o: %.cpp
	$(CC) $(INCLUDE) $(CCFLAGS) $(CXXFLAGS) -fPIC -o $@ -c $<

$(OBJ_FILES_CU): $(OBJ_DIR)/%.o: %.cu
	mkdir -p $(OBJ_DIR)/
	$(NVCC) $(INCLUDE) $(CXXFLAGS) $(CCFLAGS) $(NVCCFLAGS) -lineinfo $(GENCODE_FLAGS) -Xcompiler -fPIC -c $< -o $@

$(TARGET): $(OBJ_FILES_CU) $(OBJ_FILES_CPP)
	$(CC) $(CCFLAGS) $(CXXFLAGS) -o $@ $^ $(LIBRARIES) $(LIBRARY_FILES)
	@echo

clean:
	@rm -rf $(OBJ_DIR) $(TARGET)
