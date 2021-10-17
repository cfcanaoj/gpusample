NVFC=nvfortran

################
# phony
################

.PHONY: all clean dataclean
exes  = sample1.cpux sample1.gpux 
exes += sample2.cpux sample2.gpux 
all: $(exes)

clean:
	rm *~ *.o *.mod *.gpux *.cpux 
dataclean:
	rm job_sample*.o* job_sample*.e* sample*.gpulog  sample*.cpulog
################
# sample1
################
sample1.cpux: sample1.f90
	$(NVFC) $< -o $@

sample1.gpux: sample1.cuf
	$(NVFC) -Mcuda  $< -o $@

################
# sample2
################
sample2.cpux: sample2.f90
	$(NVFC) $< -o $@

sample2.gpux: sample2.cuf
	$(NVFC) -Mcuda  $< -o $@
