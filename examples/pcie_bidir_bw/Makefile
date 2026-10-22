NVCC ?= nvcc
# Adjust for your GPU, e.g. 90 (Hopper), 80 (Ampere A100), 89 (Ada).
ARCH ?= 90
NVCCFLAGS := -O3 -std=c++17 -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

pcie_bidir_bw: pcie_bidir_bw.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: run clean
run: pcie_bidir_bw
	./pcie_bidir_bw

clean:
	rm -f pcie_bidir_bw
