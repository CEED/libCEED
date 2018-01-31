      program test

      integer ceed,err
      integer x,y1
      integer r,rqst
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

      call ceedvectorsetarray(x,0,1,a,err)

      do i=1,ne
        ind(2*i-1)=i
        ind(2*i  )=i+1
      enddo

      call ceedelemrestrictioncreate(ceed,ne,2,ne+1,0,1,ind(1),r,err)

      call ceedvectorcreate(ceed,2*ne,y1,err);
      call ceedvectorsetarray(y1,0,0,%val(0),err);
      call ceedelemrestrictionapply(r,0,1,0,x,y1,rqst,err)

      call ceedvectorgetarrayread(y1,0,yy,err)
      do i=1,ne*2
        diff=10.0+(i-1)/2-yy(i)
        if (abs(diff) > 1.0D-5) then
          write(*,*) 'Error in restricted array y(',i,')='
        endif
      enddo
      call ceedvectorrestorearrayread(y1,yy,err)

      call ceedvectordestroy(y1,err)
      call ceedvectordestroy(x,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
