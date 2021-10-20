# Sample code for GPU computing 

## Sample1
This example shows how GPU works.
The code just calculate the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

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
