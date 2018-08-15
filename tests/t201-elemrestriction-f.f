c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,y
      integer r

      integer ne
      parameter(ne=3)

      real*8 a(2*ne)
      real*8 yy(2*ne)
      real*8 diff

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,2*ne,x,err)

      do i=1,2*ne
        a(i)=10+i-1
      enddo

      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,err)

      call ceedelemrestrictioncreateidentity(ceed,ne,2,2*ne,1,r,err)

      call ceedvectorcreate(ceed,2*ne,y,err);
      call ceedvectorsetarray(y,ceed_mem_host,ceed_copy_values,%val(0),
     $  err);
      call ceedelemrestrictionapply(r,ceed_notranspose,
     $  ceed_notranspose,x,y,ceed_request_immediate,err)

      call ceedvectorgetarrayread(y,ceed_mem_host,yy,err)
      do i=1,ne*2
        diff=10+i-1-yy(i)
        if (abs(diff) > 1.0D-15) then
          write(*,*) 'Error in restricted array y(',i,')=',yy(i)
        endif
      enddo
      call ceedvectorrestorearrayread(y,yy,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
