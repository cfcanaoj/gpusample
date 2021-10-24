#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE (64)

int main(int argc, char**argv){
  clock_t start,stop;
  
  int i;
  const int ishow=16;
  
  printf("CPU:\n");
 
  double *datah;
  datah = (double*)malloc(sizeof(double)*SIZE);
  
  for(i=0;i<SIZE;i++) datah[i] = i;

  printf("data: "); for(i=0;i<ishow;i++) printf(" %.2f",datah[i]); printf("...\n");
  
  double sum;
  sum=0.0e0;
  for(i=0;i<SIZE;i++){
    sum = sum + datah[i];
  }
  
  printf("Numerical result: "); printf(" %.2f",sum); printf("\n");

  sum = SIZE*(SIZE-1)/2;
  printf("Analytic  result: "); printf(" %.2f",sum); printf("\n");
  
  return 0;
}
