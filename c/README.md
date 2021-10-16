# Sample code for GPU computing 

## Sample1
How to complie and run the sample codes.

	cd c
	module load cuda-toolkit
	make
	./sample1_cpu.exe > sample1.cpulog
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog
	
# References

-[大島聡史, これからの並列計算のためのGPGPU連載講座（II）](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
