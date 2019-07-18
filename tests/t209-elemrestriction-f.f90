!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer mult
      integer r
      integer i
      integer*8 moffset

      integer ne
      parameter(ne=3)

      integer*4 ind(4*ne)
      real*8 mm(3*ne+1)
      integer offset
      real*8 diff

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,3*ne+1,mult,err)
      call ceedvectorsetvalue(mult,0.d0,err);

      do i=1,ne
        ind(4*i-3)=3*i-3
        ind(4*i-2)=3*i-2
        ind(4*i-1)=3*i-1
        ind(4*i-0)=3*i-0
      enddo
      call ceedelemrestrictioncreate(ceed,ne,4,3*ne+1,1,ceed_mem_host,&
     & ceed_use_pointer,ind,r,err)

      call ceedelemrestrictiongetmultiplicity(r,mult,err)

      call ceedvectorgetarrayread(mult,ceed_mem_host,mm,moffset,err)
      do i=1,3*ne+1
        if(i > 1 .and. i < 3*ne+1 .and. mod(i-1,3)==0) then
          offset = 1
        else
          offset = 0
        endif
        diff=1+offset-mm(i+moffset)
        if (abs(diff) > 1.0D-15) then
          write(*,*) 'Error in multiplicity vector: mult(',i,')=',mm(i+moffset)
        endif
      enddo
      call ceedvectorrestorearrayread(mult,mm,moffset,err)

      call ceedvectordestroy(mult,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
