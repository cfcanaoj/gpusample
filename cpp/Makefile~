NVCC=nvcc

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
sample1.cpux: sample1.c
	$(NVCC) $< -o $@

sample1.gpux: sample1.cu
	$(NVCC) $< -o $@
