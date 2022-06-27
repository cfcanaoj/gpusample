NVFC=nvfortran
NVFCOPT= -Minfo=accel -acc

################
# phony
################

.PHONY: all clean dataclean
exes  = sample1.cpux sample1.gpux 
exes  = sample2.cpux sample2.gpux 

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

sample1.gpux: sample1.f90
	$(NVFC) $(NVFCOPT) $< -o $@

################
# sample2
################
sample2.cpux: sample2.f90
	$(NVFC) $< -o $@

sample2.gpux: sample2.f90
	$(NVFC) $(NVFCOPT) $< -o $@

################
# sample3
################
sample3.cpux: sample3.f90
	$(NVFC) $< -o $@

sample3.gpux: sample3.cuf
	$(NVFC) $(NVFCOPT) $< -o $@

################
# sample4
################
sample4.cpux: sample4.f90
	$(NVFC) $< -o $@

sample4.gpux: sample4.cuf
	$(NVFC) $(NVFCOPT) $< -o $@

