#include <iostream>
#include <cstdlib>

#define SIZE 256

int main(int argc, char**argv){
  int i;
  printf("CPU:\n");
  srand(0);
  
  float *h_InA, *h_InB;
  h_InA = (float*)malloc(sizeof(float)*SIZE);
  h_InB = (float*)malloc(sizeof(float)*SIZE);

  for(i=0;i<SIZE;i++) h_InA[i] = (float)(rand()%10)/10.0f;
  for(i=0;i<SIZE;i++) h_InB[i] = (float)(rand()%10)/10.0f; 

  printf("InA: "); for(i=0;i<SIZE;i++) printf(" %.2f",h_InA[i]); printf("\n");
  printf("InB: "); for(i=0;i<SIZE;i++) printf(" %.2f",h_InB[i]); printf("\n");

  float *h_Out;
  h_Out = (float*)malloc(sizeof(float)*SIZE);

  for(i=0;i<SIZE;i++) 
    h_Out[i] = h_InA[i]+ h_InB[i];
  
  printf("Out: "); for(i=0;i<SIZE;i++) printf(" %.2f",h_Out[i]); printf("\n");
  
  return 0;
}
