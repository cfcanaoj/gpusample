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

program main
  use cuda_kernel
  implicit none
  real(4)::tbgn,tend
  integer:: i
  integer, parameter:: ishow=256
  integer, parameter:: SIZE=2048*512
  integer:: n
  integer, parameter:: nite=1000
  
  real(4),dimension(SIZE):: h_InA,h_InB,h_Out
  real(4),dimension(SIZE),device:: d_InA,d_InB,d_Out
  integer,dimension(2) :: seed
  real(8) rnum
  integer :: istat
  write(6,*) "GPU:\n"

  seed(1)=1
  seed(2)=1
  call random_seed(PUT=seed(1:2))
  istat= cudaSetDevice(0)
  do i=1, SIZE
     call random_number(rnum)
     h_InA(i) = rnum
  enddo
  do i=1, SIZE
     call random_number(rnum)
     h_InB(i) = rnum
  enddo

  write(6,*)"InA: "
  do i=1,ishow
     write(6,"(1x,f4.2)",advance='no') h_InA(i)
  enddo
  write(6,*) ""

  write(6,*)"InB: "
  do i=1,ishow
     write(6,"(1x,f4.2)",advance='no') h_InB(i)
  enddo
  write(6,*) ""
  
  istat= cudaMemcpy(d_InA, h_InA, SIZE)! attributes are specified in the declaration
  istat= cudaMemcpy(d_InB, h_InB, SIZE)! attributes are specified in the declaration

  
  call cpu_time(tbgn)
  do n=1, nite
     call arrayadd <<< 16,16 >>> (d_Out, d_InA, d_InB)
  enddo
  call cpu_time(tend)
  istat= cudaMemcpy(h_Out, d_Out, SIZE)! attributes are specified in the declaration
 
 write(6,*)"Out: "
  do i=1,ishow
     write(6,"(1x,f4.2)",advance='no') h_Out(i)
  enddo
  write(6,*) ""
  
  write(6,"(a6,(1x,f10.2,a5))")"Time: ",(tend-tbgn)*1.0e3," [ms]"
end program
