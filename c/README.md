# Sample code for GPU computing 

## Sample1
The code just calculates the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

How to compile and run the sample codes is shown as follows.

	cd c
	module load cuda-toolkit
	make
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog

### References
- [大島聡史, これからの並列計算のためのGPGPU連載講座（II）](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
- [青木 尊之 額田 彰,はじめてのCUDAプログラミング](http://www.kohgakusha.co.jp/support/cuda/index.html)

## Sample2
This example shows how GPU calculates fast. The calculation is basically the same as the sample1.

How to compile and run the sample codes is shown as follows.

	cd c
	module load cuda-toolkit
	make
	qsub pbs_sample2.sh
	cat sample2.cpulog
	cat sample2.gpulog

## Sample3
This example shows how 2D array is treated. We solve 2D poission equation by Jacobi method.  
![\begin{align*}
  \frac{\partial^2 \Phi}{\partial ^2x}
+\frac{\partial^2 \Phi}{\partial ^2y}
=4\pi G \rho
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A++%5Cfrac%7B%5Cpartial%5E2+%5CPhi%7D%7B%5Cpartial+%5E2x%7D%0A%2B%5Cfrac%7B%5Cpartial%5E2+%5CPhi%7D%7B%5Cpartial+%5E2y%7D%0A%3D4%5Cpi+G+%5Crho%0A%5Cend%7Balign%2A%7D%0A)

In the method, the gravitational potential is obtained by the following iterative procedure.
![\begin{align*}
\Phi^{n+1}_{i,j}=
\left(
  \Phi^{n}_{i+1,j}+\Phi^{n}_{i-1,j}
+\Phi^{n}_{i,j+1}+\Phi^{n}_{i,j-1}
-4\pi G \rho_{i,j}h^2
\right)/4
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5CPhi%5E%7Bn%2B1%7D_%7Bi%2Cj%7D%3D%0A%5Cleft%28%0A++%5CPhi%5E%7Bn%7D_%7Bi%2B1%2Cj%7D%2B%5CPhi%5E%7Bn%7D_%7Bi-1%2Cj%7D%0A%2B%5CPhi%5E%7Bn%7D_%7Bi%2Cj%2B1%7D%2B%5CPhi%5E%7Bn%7D_%7Bi%2Cj-1%7D%0A-4%5Cpi+G+%5Crho_%7Bi%2Cj%7Dh%5E2%0A%5Cright%29%2F4%0A%5Cend%7Balign%2A%7D%0A)

How to compile and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit
	make
	qsub pbs_sample3.sh
	cat sample3.cpulog
	cat sample3.gpulog
	
Check the graviatational potential by gnuplot. Follow the instruction in analyis serever.
	
	cd /gwork0/<username>/gpusample/c
	gnuplot
	plot "xy-gpu.dat" u 1:2:4 w l
	plot "xy-cpu.dat" u 1:2:4 w l
	

# References
- [青山龍美, GPU チュートリアル CUDA篇](https://hpc-phys.kek.jp/workshop/workshop181201.html)
