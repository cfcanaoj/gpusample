#include <iostream>
#include <cstdlib>
#include <ctime>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

__global__ void sum_wrong(const double* g_x, double* g_o) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	g_o[0] += g_x[i];
}

template <int N>
__global__ void sum_correct(const double* g_x, double* g_o) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	__shared__ double s_x[N];
	s_x[tid] = (i < N) ? g_x[i] : double{};
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

const int SIZE=64;

int main(int argc, char**argv){
  int i;
  const int ishow=16;
  thrust::  host_vector<double> datah(SIZE);
  thrust::device_vector<double> datad(SIZE);
  thrust::  host_vector<double> sumh(1);
  thrust::device_vector<double> sumd(1);
  
  double sum;
  
  const int threads = SIZE;
  const int blocks  = 1;
  
  printf("GPU:\n");

  for(i=0;i<SIZE;i++) datah[i] = i;
  		      
  printf("data: "); for(i=0;i<ishow;i++) printf(" %.2f",datah[i]); printf("...\n");
  sumh[0]=0.0e0;
  thrust::copy(datah.begin(),datah.end(),datad.begin());
  thrust::copy( sumh.begin(), sumh.end(), sumd.begin());
  cudaDeviceSynchronize();
  sum_wrong <<< blocks,threads>>>(raw_pointer_cast(datad.data())
				 ,raw_pointer_cast( sumd.data()));
  
  thrust::copy(sumd.begin(),sumd.end(),sumh.begin());
  cudaDeviceSynchronize();
  printf("Numerical result(1): "); printf(" %.2f\n",sumh[0]);

  thrust::copy( sumh.begin(), sumh.end(), sumd.begin());
  cudaDeviceSynchronize();
  sum_correct<threads> <<< blocks,threads>>>(raw_pointer_cast(datad.data())
					    ,raw_pointer_cast( sumd.data()));
  
  thrust::copy(sumd.begin(),sumd.end(),sumh.begin());
  cudaDeviceSynchronize();
  printf("Numerical result(2): "); printf(" %.2f\n",sumh[0]);

  sum = SIZE*(SIZE-1)/2;
  printf("Analytic     result: "); printf(" %.2f\n",sum);
  
  
  return 0;
}
