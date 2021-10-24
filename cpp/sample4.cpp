#include <iostream>
#include <cstdlib>
#include <ctime>

#include <vector>

const int SIZE=64;

int main(int argc, char**argv){
  int i;
  const int ishow=16;
  double sum;
  std::vector<double> datah(SIZE);
  
  printf("CPU:\n");

  for(i=0;i<SIZE;i++) datah[i] = i;

  printf("data: "); for(i=0;i<ishow;i++) printf(" %.2f",datah[i]); printf("...\n");
  
  sum=0.0;
  for(i=0;i<SIZE;i++){
    sum = sum + datah[i];
  }
  
  printf("Numerical Result: "); printf(" %.2f\n",sum);
  
  sum = SIZE*(SIZE-1)/2;
  printf("Analytic  Result: "); printf(" %.2f\n",sum);
  
  
  return 0;
}
