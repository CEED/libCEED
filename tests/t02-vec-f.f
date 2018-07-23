c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,n
      real*8 a(10)
      real*8 b(10)
      real*8 diff
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n = 10

      call ceedvectorcreate(ceed,n,x,err)

      do i=1,10
        a(i)=0
      enddo

      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,err)
      call ceedvectorgetarray(x,ceed_mem_host,b,err)
      b(3) = -3.14
      call ceedvectorrestorearray(x,b,err)
      diff=a(3)+3.14
      if (abs(diff)>1.0D-15) then
        write(*,*) 'Error writing array a(3)=',a(3)
      endif

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
