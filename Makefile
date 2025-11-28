VERILATOR = verilator
NVCC      = nvcc

TOPNAME   = LOG2FP32

VERILATOR_FLAGS = -MMD --build -cc --x-assign fast --x-initial fast --noassert --quiet-exit --trace --trace-fst

BUILD_DIR = build
OBJ_DIR   = $(BUILD_DIR)/obj_dir
TARGET    = $(BUILD_DIR)/$(TOPNAME)_sim

VSRC      = rtl/$(TOPNAME).sv
CSRC      = sim-verilator/$(TOPNAME).cpp
CUDA_SRC  = sim-verilator/$(TOPNAME)_cuda.cu
SCALA_SRC = src/scala/$(TOPNAME).scala

CUDA_OBJ  = $(BUILD_DIR)/$(TOPNAME)_cuda.o

# Auto-detect CUDA availability
CUDA_AVAILABLE := $(shell which nvcc > /dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(CUDA_AVAILABLE), 1)
	CUDA_PATH ?= /opt/cuda
	CXXFLAGS = -D__USE_GPU_REF__
	LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart $(abspath $(CUDA_OBJ))
	VERILATOR_FLAGS += -CFLAGS "$(CXXFLAGS)" -LDFLAGS "$(LDFLAGS)"
endif

$(shell mkdir -p $(BUILD_DIR))

.DEFAULT_GOAL := run

$(VSRC): $(SCALA_SRC)
	./mill --no-server $(TOPNAME).run

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) -use_fast_math -c $< -o $@

$(TARGET): $(VSRC) $(CSRC)
	@mkdir -p $(OBJ_DIR)
ifeq ($(CUDA_AVAILABLE), 1)
	@$(MAKE) $(CUDA_OBJ)
endif
	$(VERILATOR) $(VERILATOR_FLAGS) $(VSRC) $(CSRC) -Mdir $(OBJ_DIR) --exe -o $(abspath $(TARGET))

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

init:
	git submodule update --init --recursive --progress

.PHONY: run clean init
