#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE (2048*512)

int main(int argc, char**argv){
  clock_t start,stop;
  
  int i;
  const int ishow=256;
  int n;
  const int nite=1000;
  
  printf("CPU:\n");
  srand(0);

 
  float *h_InA, *h_InB, *h_Out;
  h_InA = (float*)malloc(sizeof(float)*SIZE);
  h_InB = (float*)malloc(sizeof(float)*SIZE);
  h_Out = (float*)malloc(sizeof(float)*SIZE);

  for(i=0;i<SIZE;i++) h_InA[i] = (float)(rand()%10)/10.0f;
  for(i=0;i<SIZE;i++) h_InB[i] = (float)(rand()%10)/10.0f; 

  printf("InA: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InA[i]); printf("\n");
  printf("InB: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InB[i]); printf("\n");

  start=clock();
  for(n=0;n<nite;n++){
  for(i=0;i<SIZE;i++){
    h_Out[i] = h_InA[i]+ h_InB[i];
  }
  }
  stop=clock();
  
  printf("Out: "); for(i=0;i<ishow;i++) printf(" %.2f",h_Out[i]); printf("\n");

  printf("Time: %.2f [ms]\n",(double)(stop-start));
  
  return 0;
}
