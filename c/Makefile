NVCC=nvcc

################
# phony
################

.PHONY: all clean dataclean
exes  = sample1.cpux sample1.gpux
exes += sample2.cpux sample2.gpux
exes += sample3.cpux sample3.gpux
exes += sample4.cpux sample4.gpux

all: $(exes)

clean:
	rm *~ *.o *.gpux *.cpux
dataclean:
	rm job_sample*.o* job_sample*.e* sample*.gpulog  sample*.cpulog

################
# sample1
################
sample1.cpux: sample1.c
	$(NVCC) $< -o $@

sample1.gpux: sample1.cu
	$(NVCC) $< -o $@

################
# sample2
################
sample2.cpux: sample2.c
	$(NVCC) $< -o $@

sample2.gpux: sample2.cu
	$(NVCC) $< -o $@

################
# sample3
################
sample3.cpux: sample3.c
	$(NVCC) $< -o $@

sample3.gpux: sample3.cu
	$(NVCC) $< -o $@

################
# sample4
################
sample4.cpux: sample4.c
	$(NVCC) $< -o $@

sample4.gpux: sample4.cu
	$(NVCC) $< -o $@
