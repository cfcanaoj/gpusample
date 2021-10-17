# Sample code for GPU computing 

## Sample1
This example shows how GPU works.
How to complie and run the sample codes.

	cd c
	module load cuda-toolkit
	make
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog

## Sample2
This example shows how GPU calculate fast.
How to complie and run the sample codes.

	cd c
	module load cuda-toolkit
	make
	qsub pbs_sample2.sh
	cat sample2.cpulog
	cat sample2.gpulog


# References

- [大島聡史, これからの並列計算のためのGPGPU連載講座（II）](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
- [青木 尊之 額田 彰,はじめてのCUDAプログラミング](http://www.kohgakusha.co.jp/support/cuda/index.html)
