# Sample code for GPU computing 

## Sample1
This example shows how GPU works.
The code just calculate the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

How to complie and run the sample codes is shown as follows.

	cd fortran
	module load nvhpc
	make
	qsub pbs_sample1.sh
	cat sample1.cpulog
	cat sample1.gpulog

## Sample2
This example shows how GPU calculate fast.

How to complie and run the sample codes is shown as follows.

	cd fortran
	module load nvhpc
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
	
	cd fortran
	module load nvhpc
	make
	qsub pbs_sample3.sh
	cat sample3.cpulog
	cat sample3.gpulog
	
# References
- [青山龍美, GPU チュートリアル CUDA篇](https://hpc-phys.kek.jp/workshop/workshop181201.html)
- [implicit_none, CUDA Fortran入門](https://qiita.com/implicit_none/items/8229d1931cd236d62ca9)
