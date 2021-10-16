#include <stdio.h>
#include <stdlib.h>

#define SIZE 256

int main(int argc, char**argv){
  int i;
  printf("GPU:\n");
  srand(0);
  
  float *h_InA, *h_InB;
  h_InA = (float*)malloc(sizeof(float)*SIZE);
  h_InB = (float*)malloc(sizeof(float)*SIZE);

  for(i=0;i<SIZE;i++) h_InA[i] = (float)(rand()%10)/10.0f;
  for(i=0;i<SIZE;i++) h_InB[i] = (float)(rand()%10)/10.0f; 

  return 0;
}
