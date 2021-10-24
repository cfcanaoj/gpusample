#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sum_wrong(const double* g_x, double* g_o) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	g_o[0] += g_x[i];
}

__global__ void sum_correct(const double* g_x, double* g_o, unsigned int N) {
        extern __shared__ double s_x[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	s_x[tid] = (i < N) ? g_x[i] : 0.0e0;
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		if ((tid % (2 * s)) == 0) {
			s_x[tid] += s_x[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
		g_o[blockIdx.x] = s_x[0];
}

#define SIZE (64)

int main(int argc, char**argv){
  int i;
  const int ishow=16;
  double *datah;
  double *sumh;
  
  datah = (double*)malloc(sizeof(double)*SIZE);
  sumh  = (double*)malloc(sizeof(double)*1);
  
  double *datad;
  double *sumd;
  
  cudaMalloc((void**)&datad, sizeof(double)*SIZE);
  cudaMalloc((void**)&sumd, sizeof(double)*1);
  
  const int threads = SIZE;
  const int blocks  = 1;
  
  int shared_mem_size = 2 * threads * sizeof(double);
  printf("GPU:\n");

  for(i=0;i<SIZE;i++) datah[i] = i;
  printf("data: "); for(i=0;i<ishow;i++) printf(" %.2f",datah[i]); printf("...\n");
  
  sumh[0]=0.0e0;
  cudaMemcpy(datad, datah, sizeof(double)*SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy( sumd,  sumh, sizeof(double)*1   , cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  sum_wrong <<< blocks,threads>>>(datad,sumd);
  
  cudaMemcpy( sumh,  sumd, sizeof(double)*1   , cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("Numerical result(1): "); printf(" %.2f\n",sumh[0]);

  sumh[0]=0.0e0;
  cudaMemcpy( sumd,  sumh, sizeof(double)*1   , cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  sum_correct <<< blocks,threads,shared_mem_size>>>(datad,sumd,SIZE);
  
  cudaMemcpy( sumh,  sumd, sizeof(double)*1   , cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("Numerical result(2): "); printf(" %.2f\n",sumh[0]);

  sumh[0] = SIZE*(SIZE-1)/2;
  printf("Analytic     result: "); printf(" %.2f\n",sumh[0]);
  
  return 0;
}
