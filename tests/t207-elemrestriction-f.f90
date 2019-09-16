!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,y
      integer r
      integer i,n,mult
      integer*8 aoffset,yoffset

      integer ne
      parameter(ne=3)

      integer*4 ind(2*ne)
      real*8 a(2*(ne*2))
      real*8 yy(2*(ne+1))
      real*8 diff

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,2*(ne*2),x,err)

      do i=0,ne-1
        do n=1,2
          a(i*4+n)=10+(2*i+n)/2
          a(i*4+n+2)=20+(2*i+n)/2
        enddo
      enddo

      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

      do i=1,ne
        ind(2*i-1)=i-1
        ind(2*i  )=i
      enddo

      call ceedelemrestrictioncreate(ceed,ne,2,ne+1,2,&
     & ceed_mem_host,ceed_use_pointer,ind,r,err)

      call ceedvectorcreate(ceed,2*(ne+1),y,err);
      call ceedvectorsetvalue(y,0.d0,err);
      call ceedelemrestrictionapply(r,ceed_transpose,ceed_transpose,x,y,&
     & ceed_request_immediate,err)

      call ceedvectorgetarrayread(y,ceed_mem_host,yy,yoffset,err)
      do i=0,ne
        if (i > 0 .and. i < ne) then
          mult = 2
        else
          mult = 1
        endif
        diff=(10+i)*mult-yy(2*i+1+yoffset)
        if (abs(diff) > 1.0D-15) then
! LCOV_EXCL_START
          write(*,*) 'Error in restricted array y(',2*i+1,')=',&
     &     yy(2*i+1+yoffset),'!=',(10+i)*mult
! LCOV_EXCL_STOP
        endif
        diff=(20+i)*mult-yy(2*i+2+yoffset)
        if (abs(diff) > 1.0D-15) then
! LCOV_EXCL_START
          write(*,*) 'Error in restricted array y(',2*i+2,')=',&
     &     yy(2*i+2+yoffset),'!=',(20+i)*mult
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(y,yy,yoffset,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
