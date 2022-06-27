module grids
  implicit none
  real(8),parameter:: Ggra=1.0d0
  real(8),parameter:: rho0=1.0d0
  real(8):: pi
  real(8):: x1min,x1max
  real(8):: x2min,x2max
  data x1min / 0.0d0/
  data x1max / 1.0d0/  
  data x2min / 0.0d0/
  data x2max / 1.0d0/

  integer,parameter:: margin=1
  integer,parameter:: nx1=1024
  integer,parameter:: nx2=1024

  real(8):: dx1
  real(8),dimension(nx1+1):: x1a
  real(8),dimension(nx1):: x1b,dvl1a
  
  real(8):: dx2
  real(8),dimension(nx2+1):: x2a
  real(8),dimension(nx2):: x2b,dvl2a
  
  real(8),dimension(nx1,nx2):: rho,gp
  
  integer,parameter:: l1=1
  integer,parameter:: l2=1
  real(8),parameter:: persymx1=-1.0
  real(8),parameter:: persymx2=-1.0

  integer,parameter:: itemax=300
  
end module grids

program main
  implicit none
  real(4)::tbgn,tend
  write(6,*) "CPU:\n"
  call InitializeVariables
  call SetupGrids
  call SetupDensity
  call cpu_time(tbgn)
  call GetGravitationalPotential
  call cpu_time(tend)
  call OutputData
  write(6,"(a6,(1x,f10.2,a5))")"Time: ",(tend-tbgn)*1.0e3," [ms]"
end program

subroutine InitializeVariables
  implicit none
  
  return
end subroutine InitializeVariables

subroutine SetupGrids
  use grids
  implicit none
  integer:: i,j
  dx1 =(x1max-x1min)/(nx1-margin*2)
  x1a(1) = x1min -margin*dx1
  do i=2, nx1+1
     x1a(i) =x1a(i-1) + dx1
  enddo
  do i=1, nx1
       x1b(i) = 0.5d0*(x1a(i) + x1a(i))
     dvl1a(i) = dx1
  enddo

  dx2 =(x2max-x2min)/(nx2-margin*2)
  x2a(1) =x2min -margin*dx2
  do j=2, nx2+1
     x2a(j) =x2a(j-1) + dx2
  enddo
  do j=1, nx2    
       x2b(j) = 0.5d0*(x2a(j) + x2a(j))
     dvl2a(j) = dx2
  enddo

  return
end subroutine SetupGrids

subroutine SetupDensity
  use grids
  implicit none
  integer:: i,j
  pi = acos(-1.0d0)
  do j=1, nx2    
  do i=1, nx1
     rho(i,j) = rho0 * sin(l1*pi*x1b(i)/(x1max-x1min)) &
    &                * sin(l2*pi*x2b(j)/(x2max-x2min))
  enddo
  enddo

  return
end subroutine SetupDensity

subroutine GetGravitationalPotential
  use grids
  implicit none
  real(8),dimension(nx1,nx2):: gpnxt
  real(8):: a1,a2,a3
  integer:: i,j,n
  a1 = 1.0d0/dx1**2
  a2 = 1.0d0/dx2**2
  a3 = 2.0d0*(a1+a2)

!$acc data copyin(nx1,nx2,a1,a2,a3,pi,GGra,rho,gp),copyout(gp)
!$acc kernels
  do n=1,itemax
!$acc loop independent
  do j=2, nx2-1    
!$acc loop independent
  do i=2, nx1-1
     gpnxt(i,j) = ( a1*(gp(i+1,j)+gp(i-1,j)) &
                &  +a2*(gp(i,j+1)+gp(i,j-1)) &
                &  -4.0d0*pi*Ggra*rho(i,j)   &
                & )/a3
  enddo
  enddo

!$acc loop independent
  do j=2, nx2-1
!$acc loop independent
  do i=2, nx1-1
        gp(i,j) = gpnxt(i,j)
  enddo
  enddo

  do i=1, nx1
     gp(i,  1) = persymx2*gp(i,nx2-1)
     gp(i,nx2) = persymx2*gp(i,    2)
  enddo
  
  do j=1, nx2
     gp(  1,j) = persymx1*gp(nx1-1,j)
     gp(nx1,j) = persymx1*gp(    2,j)
  enddo
  enddo
!$acc end kernels
!$acc end data 
  return
end subroutine GetGravitationalPotential

subroutine OutputData
  use grids
  implicit none
  integer:: i,j
  integer,parameter:: unitout=102
  character*50,parameter:: fileout="xy-cpu.dat"
  
  open(unit=unitout,file=fileout,status='replace',form='formatted')
  do j=1, nx2    
  do i=1, nx1
     write(unitout,"(1x,4(E15.6e3))") x1b(i), x2b(j),rho(i,j),gp(i,j)
  enddo
  write(unitout,*) ""
  enddo
  close(unitout)

  return
end subroutine OutputData
