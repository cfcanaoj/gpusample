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
  integer, parameter:: SIZE=256
  integer:: i
  real(4),dimension(SIZE):: h_InA,h_InB,h_Out
  real(4),dimension(SIZE),device:: d_InA,d_InB,d_Out
  integer,dimension(2) :: seed
  real(8) rnum
  integer :: istat
  write(6,*) "GPU:\n"

  seed(1)=1
  seed(2)=1
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
     write(6,"(1x,f4.2)",advance='no') h_InA(i)
  enddo
  write(6,*) ""

  write(6,*)"InB: "
  do i=1,SIZE
     write(6,"(1x,f4.2)",advance='no') h_InB(i)
  enddo
  write(6,*) ""
  
  istat= cudaMemcpy(d_InA, h_InA, SIZE)! attributes are specified in the declaration
  istat= cudaMemcpy(d_InB, h_InB, SIZE)! attributes are specified in the declaration

  call arrayadd <<< 16,16 >>> (d_Out, d_InA, d_InB)
  istat= cudaMemcpy(h_Out, d_Out, SIZE)! attributes are specified in the declaration
 
 write(6,*)"Out: "
  do i=1,SIZE
     write(6,"(1x,f4.2)",advance='no') h_Out(i)
  enddo
  write(6,*) ""
  
end program
