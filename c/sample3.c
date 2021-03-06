#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void InitializeVariables();
void SetupGrids();
void SetupDensity();
void GetGravitationalPotential();
void OutputData();

#define nx1 1024
#define nx2 1024

const double Ggra=1.0e0;
const double rho0=1.0e0;
double pi;
double x1min,x1max;
double x2min,x2max;

const int margin=1;

double   dx1;
double*   x1a;
double*   x1b;
double* dvl1a;

double   dx2;
double*   x2a;
double*   x2b;
double* dvl2a;
  
double* rho;
double*  gp;

const int l1=1;
const int l2=1;
const double persymx1=-1.0;
const double persymx2=-1.0;

const int itemax=300;

int main(int argc, char**argv){
  clock_t start,stop;
  printf("CPU:\n");
  InitializeVariables();
  SetupGrids();
  SetupDensity();
  start=clock();
  GetGravitationalPotential();
  stop=clock();
  OutputData();
  printf("Time: %.2f [ms]\n",(double)(stop-start));
}

void InitializeVariables(){
    x1a = (double*)malloc(sizeof(double)*(nx1+1));
    x1b = (double*)malloc(sizeof(double)*(nx1));
  dvl1a = (double*)malloc(sizeof(double)*(nx1));
  
    x2a = (double*)malloc(sizeof(double)*(nx2+1));
    x2b = (double*)malloc(sizeof(double)*(nx2));
  dvl2a = (double*)malloc(sizeof(double)*(nx2));
  
    rho = (double*)malloc(sizeof(double)*(nx1*nx2));
     gp = (double*)malloc(sizeof(double)*(nx1*nx2));
}
  
void SetupGrids(){
  int i,j;
  x1min = 0.0e0;
  x1max = 1.0e0;  
  x2min = 0.0e0;
  x2max = 1.0e0;

  dx1 =(x1max-x1min)/(nx1-margin*2);
  x1a[0] = x1min -margin*dx1;
  for(i=1;i<=nx1;i++){
    x1a[i] =x1a[i-1] + dx1;
  }
  for(i=0;i<nx1;i++){
    x1b[i] = 0.5e0*(x1a[i] + x1a[i]);
  dvl1a[i] = dx1;
  }

  dx2 =(x2max-x2min)/(nx2-margin*2);
  x2a[0] =x2min -margin*dx2;
  for(j=1;j<=nx2;j++){
    x2a[j] =x2a[j-1] + dx2;
  }
  for(j=0;j<nx2;j++){
      x2b[j] = 0.5e0*(x2a[j] + x2a[j]);
    dvl2a[j] = dx2;
  }

}
  
void SetupDensity(){
  int i,j;
  pi =atan2(0.0,-1.0);
  for(j=0;j<nx2;j++){
  for(i=0;i<nx1;i++){
       rho[j*nx1+i] = rho0 * sin(l1*pi*x1b[i]/(x1max-x1min))
                           * sin(l2*pi*x2b[j]/(x2max-x2min));
  }
  }
}

void GetGravitationalPotential(){
  double* gpnxt;
  double  a1,a2,a3;
  int  i,j,n;
  a1 = 1.0e0/dx1/dx1;
  a2 = 1.0e0/dx2/dx2;
  a3 = 2.0e0*(a1+a2);
  
  gpnxt = (double*)malloc(sizeof(double)*(nx1*nx2));
  for(n=0;n<itemax;n++){
			
  for(j=1;j<nx2-1;j++){
  for(i=1;i<nx1-1;i++){
     gpnxt[j*nx1+i] = ( a1*(gp[    j*nx1+i+1]+gp[    j*nx1+i-1]) 
		       +a2*(gp[(j+1)*nx1+i  ]+gp[(j-1)*nx1+i  ]) 
                              -4.0e0*pi*Ggra*rho[    j*nx1+i  ]   
		      )/a3;

  }
  }

  for(j=1;j<nx2-1;j++){
  for(i=1;i<nx1-1;i++){
     gp[j*nx1+i] = gpnxt[j*nx1+i];
  }
  }
  for(i=0;i<nx1;i++){
     gp[      0*nx1+i] = persymx2*gp[(nx2-2)*nx1+i];
     gp[(nx2-1)*nx1+i] = persymx2*gp[(    1)*nx1+i];
  }
  for(j=0;j<nx2;j++){
     gp[      j*nx1+0    ] = persymx1*gp[  j*nx1+nx1-2];
     gp[      j*nx1+nx1-1] = persymx1*gp[  j*nx1+    1];
  }

  } /* n-loop*/
}

void OutputData(){
  int  i,j;
  static FILE *fp;
  const char filename[30]="xy-cpu.dat";
  fp=fopen(filename,"w");
  for(j=1;j<nx2-1;j++){
  for(i=1;i<nx1-1;i++){
    fprintf(fp," %e %e %e %e\n",x1b[i], x2b[j],rho[j*nx1+i],gp[j*nx1+i]);
  }
    fprintf(fp,"\n");
  }
  fclose(fp);
}
