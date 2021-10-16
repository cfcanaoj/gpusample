NVFC=nvfortran

################
# phony
################

.PHONY: all clean
exes = sample1.cpux sample1.gpux 
all: $(exes)

clean:
	rm *~ *.o *.gpux *.cpux

################
# sample1
################
sample1.cpux: sample1_cpu.f90
	$(NVFC) $< -o $@

sample1.gpux: sample1_gpu.f90
	$(NVFC) $< -o $@
