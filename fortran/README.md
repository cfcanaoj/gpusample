# Sample code for GPU computing 

## Sample1
How to complie and run the sample codes.

	cd cpp
	module load nvhpc
	make
	./sample1_cpu.exe > sample1.cpulog
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog
	
# References
- [青山龍美, GPU チュートリアル CUDA篇](https://hpc-phys.kek.jp/workshop/workshop181201.html)
- [implicit_none, CUDA Fortran入門](https://qiita.com/implicit_none/items/8229d1931cd236d62ca9)
