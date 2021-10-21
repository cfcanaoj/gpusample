#include <iostream>
#include <cstdlib>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#define SIZE 256

int main(int argc, char**argv){
  printf("GPU:\n");
  int i;
  const int ishow=16;
  
  cudaSetDevice(0);
  /* variables in device */
  thrust::device_vector<float> d_InA(SIZE);
  thrust::device_vector<float> d_InB(SIZE);
  thrust::device_vector<float> d_Out(SIZE);
  /* variables in host */
  thrust::  host_vector<float> h_InA(SIZE);
  thrust::  host_vector<float> h_InB(SIZE);
  thrust::  host_vector<float> h_Out(SIZE);
  
  srand(0);
  /* initialize */
  for(i=0;i<SIZE;i++) h_InA[i] = (float)(rand()%10)/10.0f;
  for(i=0;i<SIZE;i++) h_InB[i] = (float)(rand()%10)/10.0f; 

  /* confirm */
  printf("InA: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InA[i]); printf("\n");
  printf("InB: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InB[i]); printf("\n");
  
  /* transfer from host to device */
  thrust::copy(h_InA.begin(),h_InA.end(),d_InA.begin());/* d_InA = h_InA */ 
  thrust::copy(h_InB.begin(),h_InB.end(),d_InB.begin());/* d_InB = h_InB */

  cudaDeviceSynchronize();

  /* call kernel functions, specify grid and block as <<< grid, block >>> */
 
  thrust::transform(d_InA.begin(), d_InA.end(), d_InB.begin(), d_Out.begin(), thrust::plus<float>());
  
  cudaDeviceSynchronize();
 
  /* transfer from device to host */
  thrust::copy(d_Out.begin(),d_Out.end(),h_Out.begin());
 
  /* confirm */
  printf("Out: "); for(i=0;i<ishow;i++) printf(" %.2f",h_Out[i]); printf("\n");
  
  return 0;
}

