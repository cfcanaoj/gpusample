NVCC=nvcc

sample1.gpux: sample1.cu
	$(NVCC) $< -o $@

clean:
	rm *~ *.o *.gpux *.cpux

