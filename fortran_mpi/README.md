# Sample code for GPU computing 

## Sample1
How to complie and run the sample codes.

	cd cpp
	module load nvhpc
	make
	qsub pbs_sample1_cpu.sh
	qsub pbs_sample1_gpu.sh
	cat sample1-*.cpulog
	cat sample1-*.gpulog
	

# References

-[1](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
-[2](https://hpc-phys.kek.jp/workshop/workshop181201.html)
-[3](https://qiita.com/implicit_none/items/8229d1931cd236d62ca9)