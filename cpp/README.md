# Sample code for GPU computing 

## Sample1
The code just calculate the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

How to complie and run the sample codes.

	cd cpp
	module load cuda-toolkit
	make
	./sample1_cpu.exe > sample1.cpulog
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog
	
# References

-[大島聡史, これからの並列計算のためのGPGPU連載講座（II）](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
