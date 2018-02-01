      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,y1
      integer r
      parameter(ne=3)

      integer*8 ind(2*ne)
      real*8 a(ne+1)
      real*8 yy(2*ne)
      real*8 diff

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,ne+1,x,err)

      do i=1,ne+1
        a(i)=10+i
      enddo

      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,err)

      do i=1,ne
        ind(2*i-1)=i-1
        ind(2*i  )=i
      enddo

      call ceedelemrestrictioncreate(ceed,ne,2,ne+1,ceed_mem_host,
     $  ceed_use_pointer,ind,r,err)

      call ceedvectorcreate(ceed,2*ne,y1,err);
      call ceedvectorsetarray(y1,ceed_mem_host,ceed_copy_values,%val(0),
     $  err);
      call ceedelemrestrictionapply(r,ceed_notranspose,1,
     $  ceed_notranspose,x,y1,ceed_request_immediate,err)

      call ceedvectorgetarrayread(y1,ceed_mem_host,yy,err)
      do i=1,ne*2
        diff=10.0+i/2-yy(i)
        if (abs(diff) > 1.0D-5) then
          write(*,*) 'Error in restricted array y(',i,')=',yy(i)
        endif
      enddo
      call ceedvectorrestorearrayread(y1,yy,err)

      call ceedvectordestroy(y1,err)
      call ceedvectordestroy(x,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
