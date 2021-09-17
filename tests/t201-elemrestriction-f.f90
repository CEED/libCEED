!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer x,y
      integer r
      integer i,j,k

      integer ne
      parameter(ne=3)
      integer strides(3)
      integer layout(3)

      real*8 a(2*ne)
      real*8 yy(2*ne)
      real*8 diff
      integer*8 aoffset,yoffset

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,2*ne,x,err)

      do i=1,2*ne
        a(i)=10+i-1
      enddo

      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

      strides=[1,2,2]
      call ceedelemrestrictioncreatestrided(ceed,ne,2,1,2*ne,strides,r,err)

      call ceedvectorcreate(ceed,2*ne,y,err)
      call ceedvectorsetvalue(y,0.d0,err)
      call ceedelemrestrictionapply(r,ceed_notranspose,x,y,&
     & ceed_request_immediate,err)

      call ceedvectorgetarrayread(y,ceed_mem_host,yy,yoffset,err)
      call ceedelemrestrictiongetelayout(r,layout,err)
      do i=0,1
        do j=0,0
          do k=0,ne-1
            diff=yy(yoffset+i*layout(1)+j*layout(2)+k*layout(3)+1)
            diff=diff-a(i*strides(1)+j*strides(2)+k*strides(3)+1)
            if (abs(diff) > 1.0D-15) then
! LCOV_EXCL_START
             write(*,*) 'Error in restricted array y(',i,')(',j,')(',k,')=',&
     &         yy(yoffset+i*layout(1)+j*layout(2)+k*layout(3)+1)
! LCOV_EXCL_STOP
            endif
          enddo
        enddo
      enddo
      call ceedvectorrestorearrayread(y,yy,yoffset,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
