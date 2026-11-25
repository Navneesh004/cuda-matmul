# Build:   make              (detects your GPU's architecture with nvidia-smi)
#          make ARCH=sm_75   (T4)   sm_80 (A100)   sm_86 (RTX 30)   sm_89 (RTX 40 / L4)
# Test:    make test         (correctness of every kernel at awkward sizes)

NVCC  ?= nvcc
# Ask the driver for the compute capability (e.g. "7.5" -> sm_75); fall back to T4.
DETECTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | grep -E '^[0-9]+\.[0-9]+$$' | tr -d '.')
ARCH  ?= sm_$(if $(DETECTED),$(DETECTED),75)
FLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas=-v
LIBS  := -lcublas

SRC  := src/main.cu
DEPS := $(wildcard src/*.cuh src/kernels/*.cuh)
BIN  := matmul

$(BIN): $(SRC) $(DEPS)
	$(NVCC) $(FLAGS) $(SRC) -o $@ $(LIBS) 2> build.log || (cat build.log; exit 1)
	@echo "built $@ for $(ARCH) (register/shared-memory usage per kernel in build.log)"

test: $(BIN)
	bash scripts/test.sh

clean:
	rm -f $(BIN) build.log

.PHONY: test clean
