NVCC=nvcc

################
# phony
################

.PHONY: all clean dataclean
exes = sample1.cpux sample1.gpux 
all: $(exes)

clean:
	rm *~ *.o *.gpux *.cpux
dataclean:
	rm job_sample*.o* job_sample*.e* sample*.gpulog  sample*.cpulog

################
# sample1
################
sample1.cpux: sample1.cpp
	$(NVCC) $< -o $@

sample1.gpux: sample1.cu
	$(NVCC) $< -o $@
