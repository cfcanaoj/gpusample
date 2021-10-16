#include <stdio.h>
#include <stdlib.h>

#define SIZE 256

__global__ void arrayadd(float *fOut, float *fInA, float *fInB){
  int id = threadldx.x + blockldx.x  * blockDim.x;
  fOut[id] = fInA[id] +fInB[id];
}

int main(int argc, char**argv){
  int i;
  printf("GPU:\n");
  srand(0);

  cudaSetDevice(0);
  
  float *h_InA, *h_InB;
  h_InA = (float*)malloc(sizeof(float)*SIZE);
  h_InB = (float*)malloc(sizeof(float)*SIZE);

  for(i=0;i<SIZE;i++) h_InA[i] = (float)(rand()%10)/10.0f;
  for(i=0;i<SIZE;i++) h_InB[i] = (float)(rand()%10)/10.0f; 

  return 0;
}
