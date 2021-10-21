#include <iostream>
#include <cstdlib>
#include <vector>

#define SIZE 256

int main(int argc, char**argv){
  int i;
  const int ishow=16;
  printf("CPU:\n");
  srand(0);
  std::vector<float> h_InA(SIZE),h_InB(SIZE),h_Out(SIZE);

  for(i=0;i<SIZE;i++) h_InA[i] = (float)(rand()%10)/10.0f;
  for(i=0;i<SIZE;i++) h_InB[i] = (float)(rand()%10)/10.0f; 

  printf("InA: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InA[i]); printf("\n");
  printf("InB: "); for(i=0;i<ishow;i++) printf(" %.2f",h_InB[i]); printf("\n");

  for(i=0;i<SIZE;i++) 
    h_Out[i] = h_InA[i]+ h_InB[i];
  
  printf("Out: "); for(i=0;i<ishow;i++) printf(" %.2f",h_Out[i]); printf("\n");
  
  return 0;
}
