program main
  implicit none
  integer, parameter:: SIZE=256
  integer:: i
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
  do i=1,SIZE
     write(6,"(1x,f4.2)",advance='no') h_InA(i)
  enddo
  write(6,*) ""

  write(6,*)"InB: "
  do i=1,SIZE
     write(6,"(1x,f4.2)",advance='no') h_InB(i)
  enddo
  write(6,*) ""
  

  do i=1, SIZE
    h_Out(i) = h_InA(i)+ h_InB(i)
 enddo
 
 write(6,*)"Out: "
  do i=1,SIZE
     write(6,"(1x,f4.2)",advance='no') h_Out(i)
  enddo
  write(6,*) ""
  
end program
