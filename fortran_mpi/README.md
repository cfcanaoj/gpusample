# Sample code for GPU computing 

## Sample1
The code just calculate the sum of the two vectors.  
![\begin{align*}
C_i=A_i+B_i
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AC_i%3DA_i%2BB_i%0A%5Cend%7Balign%2A%7D%0A%0A)

How to complie and run the sample codes.

	cd cpp
	module load nvhpc
	make
	qsub pbs_sample1_cpu.sh
	qsub pbs_sample1_gpu.sh
	cat sample1-*.cpulog
	cat sample1-*.gpulog
	
# References
- [青山龍美, GPU チュートリアル CUDA篇](https://hpc-phys.kek.jp/workshop/workshop181201.html)
- [implicit_none, CUDA Fortran入門](https://qiita.com/implicit_none/items/8229d1931cd236d62ca9)
