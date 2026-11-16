# Build lu_solve_gpu for use as a nixnan instrumentation target.
#
# -lineinfo is required for nixnan's LINE_INFO=1 source-location reporting
# (see ../../Tutorial.md, "Basic Usage").

NVCC = nvcc
ARCH = sm_86
NVCCFLAGS = -std=c++17 -g -O2 -lineinfo -arch=$(ARCH)

SRC = lu_solve_gpu.cu
EXECUTABLE_NAME = lu_solve_gpu

all: $(EXECUTABLE_NAME) lu_solve_gpu_tc lu_solver_gpu_noclamp

# FP64 solver WITHOUT the epsilon clamp (see CLAMP_STORY.md): same source as the clamp study, built -DCLAMP=0.
lu_solver_gpu_noclamp: lu_solve_gpu_clamp_study.cu
	$(NVCC) $(NVCCFLAGS) -DREAL=double -DCLAMP=0 -o $@ $<

lu_solve_gpu_tc: lu_solve_gpu_tc.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(EXECUTABLE_NAME): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(EXECUTABLE_NAME) lu_solve_gpu_tc lu_solver_gpu_noclamp

.PHONY: all clean
