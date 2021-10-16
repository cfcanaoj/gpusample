module cuda_kernel
  use cudafor
contains
  attributes(global) subroutine arrayadd(fOut,fInA,fInB)
    implicit none
    real(4),dimension(:),device::fInA,fInB
    real(4),dimension(:),device::fOut
    integer::id
    id = threadIdx%x + blockDim%x * (blockIdx%x-1)
    fOut(id) = fInA(id) + fInB(id)
  end subroutine arrayadd
end module cuda_kernel

module mpisetup
  implicit none
  include "mpif.h"
!==========================================
! Setup for MPI
!==========================================
  integer, parameter :: mreq  = 300
  integer :: stat(MPI_STATUS_SIZE,mreq)
  integer :: req(mreq)
  integer :: nprocs_w,myid_w
  integer :: ierr
contains
  subroutine InitiaizeMPI
   implicit none
!==========================================
! Initialize MPI
!==========================================
  call MPI_INIT( ierr )
  call MPI_COMM_SIZE( MPI_COMM_WORLD, nprocs_w, ierr )
  call MPI_COMM_RANK( MPI_COMM_WORLD, myid_w  , ierr )
  end subroutine InitiaizeMPI
end module mpisetup

program main
  use cuda_kernel
  use mpisetup
  implicit none
  integer, parameter:: SIZE=128
  integer:: i
  real(4),dimension(SIZE):: h_InA,h_InB,h_Out
  real(4),dimension(SIZE),device:: d_InA,d_InB,d_Out
  integer,dimension(2) :: seed
  real(8) rnum
  integer,parameter::unitout=121
  character*50::fileout
  integer :: istat

  nprocs_w=2
  call InitiaizeMPI
  write(fileout,"(a8,i3.3,a7)") "sample1-",myid_w,".gpulog"
  open (unit=unitout,file=fileout,status='replace',form='formatted')
  write(unitout,*) "GPU:\n"

  seed(1)=1
  seed(2)=1+myid_w*nprocs_w
  call random_seed(PUT=seed(1:2))
  
  do i=1, SIZE
     call random_number(rnum)
     h_InA(i) = rnum
  enddo
  do i=1, SIZE
     call random_number(rnum)
     h_InB(i) = rnum
  enddo

  write(unitout,*)"InA: "
  do i=1,SIZE
     write(unitout,"(1x,f4.2)",advance='no') h_InA(i)
  enddo
  write(unitout,*) ""

  write(unitout,*)"InB: "
  do i=1,SIZE
     write(unitout,"(1x,f4.2)",advance='no') h_InB(i)
  enddo
  write(unitout,*) ""
  
  istat= cudaMemcpy(d_InA, h_InA, SIZE)! attributes are specified in the declaration
  istat= cudaMemcpy(d_InB, h_InB, SIZE)! attributes are specified in the declaration

  call arrayadd <<< 16,16 >>> (d_Out, d_InA, d_InB)
  istat= cudaMemcpy(h_Out, d_Out, SIZE)! attributes are specified in the declaration
 
 write(unitout,*)"Out: "
  do i=1,SIZE
     write(unitout,"(1x,f4.2)",advance='no') h_Out(i)
  enddo
  write(unitout,*) ""
  
end program
