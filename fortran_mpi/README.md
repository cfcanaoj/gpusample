# Sample code for GPU computing 

## Sample1
How to complie and run the sample codes.

	cd cpp
	module load nvhpc
	make
	qsub pbs_sample1_cpu.sh
	qsub pbs_sample1_gpu.sh
	cat sample1-*.cpulog
	cat sample1-*.gpulog
	
# References
- [大島聡史, これからの並列計算のためのGPGPU連載講座（II）](https://www.cc.u-tokyo.ac.jp/public/VOL12/No2/201003gpgpu.pdf)
- [青山龍美, GPU チュートリアル CUDA篇](https://hpc-phys.kek.jp/workshop/workshop181201.html)
- [implicit_none, CUDA Fortran入門](https://qiita.com/implicit_none/items/8229d1931cd236d62ca9)
