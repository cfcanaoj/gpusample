# Sample code for GPU computing 

## Sample1
The code just calculate the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

How to complie and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit/11.0
	make
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog
	
## Sample2
This example shows how GPU calculate fast.

How to complie and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit/11.0
	make
	qsub pbs_sample2.sh
	cat sample2.cpulog
	cat sample2.gpulog
	
	
## Sample3
This example shows how 2D array is treated. We solve 2D poission equation by Jacobi Method.  
![\begin{align*}
  \frac{\partial^2 \Phi}{\partial ^2x}
+\frac{\partial^2 \Phi}{\partial ^2y}
=4\pi G \rho
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A++%5Cfrac%7B%5Cpartial%5E2+%5CPhi%7D%7B%5Cpartial+%5E2x%7D%0A%2B%5Cfrac%7B%5Cpartial%5E2+%5CPhi%7D%7B%5Cpartial+%5E2y%7D%0A%3D4%5Cpi+G+%5Crho%0A%5Cend%7Balign%2A%7D%0A)

How to complie and run the sample codes is shown as follows.

	cd cpp
	module load cuda-toolkit/11.0
	make
	qsub pbs_sample3.sh
	cat sample3.cpulog
	cat sample3.gpulog
	
Check the graviatational potential by gnuplot. Follow the instruction in analyis serever.
	
	cd /gwork0/<username>/gpusample/fortran
	gnuplot
	plot "xy-gpu.dat" u 1:2:4 w l
	plot "xy-cpu.dat" u 1:2:4 w l
	

# References

- [大島聡史, これからの並列計算のためのGPGPU連載講座（II）](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
- [Manual of thrust](https://thrust.github.io/doc/structthrust_1_1plus.html)
