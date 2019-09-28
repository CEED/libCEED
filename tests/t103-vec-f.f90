!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,y,n
      integer*8 aoffset,xoffset,yoffset
      real*8 a(10)
      real*8 xx(10)
      real*8 yy(10)
      real*8 diff
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10
      call ceedvectorcreate(ceed,n,x,err)
      call ceedvectorcreate(ceed,n,y,err)

      do i=1,10
        a(i)=10+i
      enddo

      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

      call ceedvectorgetarray(x,ceed_mem_host,xx,xoffset,err)
      call ceedvectorsetarray(y,ceed_mem_host,ceed_copy_values,xx,xoffset,err)
      call ceedvectorrestorearray(x,xx,xoffset,err)

      call ceedvectorgetarrayread(y,ceed_mem_host,yy,yoffset,err)

      do i=1,10
        diff=yy(i+yoffset)-10-i
        if (abs(diff)>1.0D-15) then
! LCOV_EXCL_START
          write(*,*) 'Error reading array y(',i,')=',yy(i+yoffset)
! LCOV_EXCL_STOP
        endif
      enddo

      call ceedvectorrestorearrayread(y,yy,yoffset,err)
      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
