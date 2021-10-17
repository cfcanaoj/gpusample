program main
  implicit none
  real(4)::tbgn,tend
  integer:: i
  integer, parameter:: ishow=256
  integer, parameter:: SIZE=2048*512
  integer:: n
  integer, parameter:: nite=1000
  
  real(4),dimension(SIZE):: h_InA,h_InB,h_Out
  integer,dimension(2) :: seed
  real(8)::rnum
  write(6,*) "CPU:\n"

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
  do i=1,ishow
     write(6,"(1x,f4.2)",advance='no') h_InA(i)
  enddo
  write(6,*) ""

  write(6,*)"InB: "
  do i=1,ishow
     write(6,"(1x,f4.2)",advance='no') h_InB(i)
  enddo
  write(6,*) ""

  call cpu_time(tbgn)
  do n=1, nite
     do i=1, SIZE
        h_Out(i) = h_InA(i)+ h_InB(i)
     enddo
  enddo
  call cpu_time(tend)
 
 write(6,*)"Out: "
  do i=1,ishow
     write(6,"(1x,f4.2)",advance='no') h_Out(i)
  enddo
  write(6,*) ""
  
  write(6,"(a6,(1x,f10.2,a5))")"Time: ",(tend-tbgn)*1.0e3," [ms]"
end program
