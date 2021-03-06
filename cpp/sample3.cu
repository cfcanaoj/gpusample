#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

void InitializeVariables();
void SetupGrids();
void SetupDensity();
void GetGravitationalPotential();
void OutputData();

/* define kernel functions */
__global__ void jacobi2D(double* gpnxt,
			 double*    gp,
			 double*   rho,
                         double a1, double a2, double a3, double c,
			 int nx1, int nx2){
  int i,j;
  i = threadIdx.x + blockIdx.x  * blockDim.x;
  j = threadIdx.y + blockIdx.y  * blockDim.y;
  
  if (  (i<=0) || (i>=nx1-1) || (j<=0) || (j>=nx2-1) ) return;
  
     gpnxt[j*nx1+i] = ( a1*(gp[    j*nx1+i+1]+gp[    j*nx1+i-1]) 
		       +a2*(gp[(j+1)*nx1+i  ]+gp[(j-1)*nx1+i  ]) 
                                          -c*rho[    j*nx1+i  ]   
		      )/a3;
}

__global__ void copygp(double* gp_to,
		       double* gp_from,
                       double persymx1,double persymx2,
		       int nx1, int nx2){
  int i,j;
  i = threadIdx.x + blockIdx.x  * blockDim.x;
  j = threadIdx.y + blockIdx.y  * blockDim.y;

  if       (i==0){
    gp_to[j*nx1+i] = persymx1*gp_from[j*nx1+(nx1-2)]; 
  }else if (i==nx1-1) {
    gp_to[j*nx1+i] = persymx1*gp_from[j*nx1+1];
  }else if (j==0) {
    gp_to[j*nx1+i] = persymx2*gp_from[(nx2-2)*nx1+i];
  }else if (j==nx2-1) {
    gp_to[j*nx1+i] = persymx2*gp_from[    (1)*nx1+i];
  }else{
    gp_to[j*nx1+i] = gp_from[j*nx1+i];
  }
}


namespace grids{
  const double Ggra=1.0e0;
  const double rho0=1.0e0;
  double pi;
  double x1min,x1max;
  double x2min,x2max;

  const int margin=1;
  const int nx1=1024;
  const int nx2=1024;

  double dx1;
  thrust::host_vector<double>  x1a(nx1+1);
  thrust::host_vector<double>  x1b(nx1),dvl1a(nx1);
  
  double dx2;
  thrust::host_vector<double>  x2a(nx2+1);
  thrust::host_vector<double>  x2b(nx2),dvl2a(nx2);
  
  thrust::  host_vector<double>   rho(nx1*nx2), gp(nx1*nx2);
  thrust::device_vector<double>  rhod(nx1*nx2),gpd(nx1*nx2);
  
  const int l1=1;
  const int l2=1;
  const double persymx1=-1.0;
  const double persymx2=-1.0;

  const int itemax=300;
};

int main(int argc, char**argv){
  clock_t start,stop;
  printf("GPU:\n");
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
}
  
void SetupGrids(){
  using namespace grids;
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
  using namespace grids;
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
  using namespace grids;
  thrust::device_vector<double>  gpnxtd(nx1*nx2);
  double  a1,a2,a3,c;
  int n;
  a1 = 1.0e0/dx1/dx1;
  a2 = 1.0e0/dx2/dx2;
  a3 = 2.0e0*(a1+a2);
  c  = 4.0*pi*Ggra;
  thrust::copy(rho.begin(),rho.end(),rhod.begin());
  thrust::copy( gp.begin(), gp.end(), gpd.begin());
    
  const int Nthread=32;
  dim3 grid(nx1/Nthread,nx2/Nthread,1), block(Nthread,Nthread,1);
    
  for(n=0;n<itemax;n++){
    jacobi2D <<< grid,block >>> (thrust::raw_pointer_cast(gpnxtd.data()),
                                 thrust::raw_pointer_cast(   gpd.data()),
                                 thrust::raw_pointer_cast(  rhod.data()),
				 a1,a2,a3,c,nx1,nx2);
    cudaDeviceSynchronize();
    copygp  <<<  grid,block >>> (thrust::raw_pointer_cast(   gpd.data()),
                                 thrust::raw_pointer_cast(gpnxtd.data()),
                                 persymx1,persymx2,nx1,nx2);
    cudaDeviceSynchronize();
  } /* n-loop*/
  
  thrust::copy(gpd.begin(),gpd.end(),gp.begin());

}

void OutputData(){
  using namespace grids;
  int  i,j;
  static FILE *fp;
  const char filename[30]="xy-gpu.dat";
  fp=fopen(filename,"w");
  for(j=1;j<nx2-1;j++){
  for(i=1;i<nx1-1;i++){
    fprintf(fp," %e %e %e %e\n",x1b[i], x2b[j],rho[j*nx1+i],gp[j*nx1+i]);
  }
    fprintf(fp,"\n");
  }
  fclose(fp);
}
