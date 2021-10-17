#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024

/* define kernel functions */
__global__ void arrayadd(float *fOut, float *fInA, float *fInB){
  int id = threadIdx.x + blockIdx.x  * blockDim.x;
  fOut[id] = fInA[id] +fInB[id];
}

int main(int argc, char**argv){
  clock_t start,stop;
  int i;
  const int ishow=256;
  int n;
  const int nite=1000;
  
  printf("GPU:\n");
  srand(0);

  cudaSetDevice(0);
  /* variables in host */
  float *h_InA, *h_InB, *h_Out;
  h_InA = (float*)malloc(sizeof(float)*SIZE);
  h_InB = (float*)malloc(sizeof(float)*SIZE);
  h_Out = (float*)malloc(sizeof(float)*SIZE);

  /* initialize */
  for(i=0;i<SIZE;i++) h_InA[i] = (float)(rand()%10)/10.0f;
  for(i=0;i<SIZE;i++) h_InB[i] = (float)(rand()%10)/10.0f; 

  /* confirm */
  printf("InA: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InA[i]); printf("\n");
  printf("InB: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InB[i]); printf("\n");

  
  /* variables in device */
  float *d_InA, *d_InB, *d_Out;
  cudaMalloc((void**)&d_InA, sizeof(float)*SIZE);
  cudaMalloc((void**)&d_InB, sizeof(float)*SIZE);
  cudaMalloc((void**)&d_Out, sizeof(float)*SIZE);

  start=clock();  
  /* transfer from host to device */
  cudaMemcpy(d_InA, h_InA, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_InB, h_InB, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Out, h_Out, sizeof(float)*SIZE, cudaMemcpyHostToDevice);

  for(n=0;n<nite;n++){
  /* call kernel functions, specify grid and block as <<< grid, block >>> */
     arrayadd<<< 16,16 >>> (d_Out,d_InA, d_InB);
  } 

  cudaDeviceSynchronize(); 
  /* transfer from device to host */
  cudaMemcpy(h_Out, d_Out, sizeof(float)*SIZE, cudaMemcpyDeviceToHost);
  stop=clock();
 
  /* confirm */
  printf("Out: "); for(i=0;i<ishow;i++) printf(" %.2f",h_Out[i]); printf("\n");
  printf("Time: %.2f [ms]\n",(double)(stop-start));
  
  return 0;
}

