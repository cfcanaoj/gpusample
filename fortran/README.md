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

-[1](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
