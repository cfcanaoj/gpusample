program main
  implicit none
  real(4)::tbgn,tend
  integer:: i
  integer, parameter:: ishow=16
  integer, parameter:: SIZE=64
  
  real(8),dimension(SIZE):: datah
  real(8):: sum
  
  write(6,*) "CPU:\n"

  
  do i=1, SIZE
     datah(i) = i
  enddo

  write(6,*,advance='no')"data: "
  do i=1,ishow
     write(6,"(1x,f8.2)",advance='no') datah(i)
  enddo
  write(6,*) ""

  sum=0.0d0
!$acc data copyin(sum,datah),copyout(sum)
!$acc parallel loop reduction(+:sum)
  do i=1, SIZE
     sum = sum + datah(i)
  enddo
!$acc end parallel
!$acc end data 
 
  write(6,*,advance='no')"Numerical result: "
  write(6,"(1x,f8.2)") sum
  sum = SIZE*(SIZE+1)/2
  write(6,*,advance='no')"Analytic  result: "
  write(6,"(1x,f8.2)") sum
 
end program
