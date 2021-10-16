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
  use mpisetup
  implicit none
!==========================================
  integer, parameter:: SIZE=128
  integer:: i
  real(4),dimension(SIZE):: h_InA,h_InB,h_Out
  integer,dimension(2) :: seed
  real(8)::rnum
  integer,parameter::unitout=121
  character*50::fileout
  nprocs_w=2
  call InitiaizeMPI

  write(fileout,"(a8,i3.3,a7)") "sample1-",myid_w,".cpulog"
  open (unit=unitout,file=fileout,status='replace',form='formatted')
  write(unitout,*) "CPU:\n"

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

  write(6,*)"InA: "
  do i=1,SIZE
     write(unitout,"(1x,f4.2)",advance='no') h_InA(i)
  enddo
  write(unitout,*) ""

  write(unitout,*)"InB: "
  do i=1,SIZE
     write(unitout,"(1x,f4.2)",advance='no') h_InB(i)
  enddo
  write(unitout,*) ""
  

  do i=1, SIZE
    h_Out(i) = h_InA(i)+ h_InB(i)
 enddo
 
 write(unitout,*)"Out: "
  do i=1,SIZE
     write(unitout,"(1x,f4.2)",advance='no') h_Out(i)
  enddo
  write(unitout,*) ""

end program
