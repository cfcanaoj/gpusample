# Sample code for GPU computing 

## Sample1
The code just calculates the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

See the code for [CPU](./sample1.f90) and [GPU](./sample1.cuf).

How to compile and run the sample codes is shown as follows.

	cd fortran
	module load nvhpc
	make
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog
	
### References
- [implicit_none, CUDA Fortran入門](https://qiita.com/implicit_none/items/8229d1931cd236d62ca9)

## Sample2
This example shows how GPU calculates fast. The calculation is basically the same as the sample1.

See the code for [CPU](./sample2.f90) and [GPU](./sample2.cuf).

How to compile and run the sample codes is shown as follows.

	cd fortran
	module load nvhpc
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

See the code for [CPU](./sample3.f90) and [GPU](./sample3.cuf).

How to compile and run the sample codes is shown as follows.
	
	cd fortran
	module load nvhpc
	make
	qsub pbs_sample3.sh
	cat sample3.cpulog
	cat sample3.gpulog
	
Check the graviatational potential by gnuplot. Follow the instruction in analyis serever.
	
	cd /gwork0/<username>/gpusample/fortran
	gnuplot
	plot "xy-gpu.dat" u 1:2:4 w l
	plot "xy-cpu.dat" u 1:2:4 w l
	
### References

- [青山龍美, GPU チュートリアル CUDA篇](https://hpc-phys.kek.jp/workshop/workshop181201.html)

## Sample4
This example shows how the summation of all components of a vector are obtained.
	
![\begin{align*}
S=\sum_i^N a_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AS%3D%5Csum_i%5EN+a_i%0A%5Cend%7Balign%2A%7D%0A)
	
See the code for [CPU](./sample4.cpp) and [GPU](./sample4.cu). This code implementation is the simplest and slowest one, so if you want to use faster code, see the references.
	
How to compile and run the sample codes is shown as follows.

	cd fortran
	module load nvhpc
	make
	qsub pbs_sample4.sh
	cat sample4.cpulog
	cat sample4.gpulog
	
### References
- [小川, CUDA Parallel Reduction](https://ipx.hatenablog.com/entry/2017/08/31/130102)
- [gyu-don, CUDAでCUDAで配列の総和を求めてみた](https://qiita.com/gyu-don/items/ef8a128fa24f6bddd342)
- [丸山直也, CUDAプログラムの最適化](http://gpu-computing.gsic.titech.ac.jp/Japanese/Lecture/2010-06-28/reduction.pdf)
- [NVIDIA HPC SDK Version 21.9 Documentation, 3.2.5 Sharad data](https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/index.html#cfref-var-attr-shared-data)
