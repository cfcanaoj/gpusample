# Sample code for GPU computing 

## Sample1
The code just calculates the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

See the code for [cpu](./sample1.cpp) and [gpu](./sample1.cu).

How to compile and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit/11.0
	make
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog
	
### References
- [大島聡史, これからの並列計算のためのGPGPU連載講座（II）](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
- [Manual of thrust](https://thrust.github.io/doc/structthrust_1_1plus.html)

## Sample2
This example shows how GPU calculates fast. The calculation is basically the same as the sample1.

See the code for [cpu](./sample2.cpp) and [gpu](./sample2.cu).

How to compile and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit/11.0
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

See the code for [cpu](./sample3.cpp) and [gpu](./sample3.cu).

How to compile and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit/11.0
	make
	qsub pbs_sample3.sh
	cat sample3.cpulog
	cat sample3.gpulog
	
Check the gravitational potential by gnuplot. Follow the instruction in the analysis server..
	
	cd /gwork0/<username>/gpusample/cpp
	gnuplot
	plot "xy-gpu.dat" u 1:2:4 w l
	plot "xy-cpu.dat" u 1:2:4 w l
	

### References
- [青山龍美, GPU チュートリアル CUDA篇](https://hpc-phys.kek.jp/workshop/workshop181201.html)
- [how to cast a 2-dimensional thrust::device_vector<thrust::device_vector<int>> to raw pointer, stack overflow](https://stackoverflow.com/questions/38056472/how-to-cast-a-2-dimensional-thrustdevice-vectorthrustdevice-vectorint-to)
	
## Sample4
This example shows how the summation of all components of a vector are obtained.

See the code for [cpu](./sample4.cpp) and [gpu](./sample4.cu). This code implementation is the simplest and slowest one, so if you want to use faster code, please refer to the reference material.
	
How to compile and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit/11.0
	make
	qsub pbs_sample4.sh
	cat sample4.cpulog
	cat sample4.gpulog
	
### References
- [小川, CUDA Parallel Reduction](https://ipx.hatenablog.com/entry/2017/08/31/130102)
- [gyu-don, CUDAでCUDAで配列の総和を求めてみた](https://qiita.com/gyu-don/items/ef8a128fa24f6bddd342)
- [丸山直也, CUDAプログラムの最適化](http://gpu-computing.gsic.titech.ac.jp/Japanese/Lecture/2010-06-28/reduction.pdf)
