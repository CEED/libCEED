c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer b
      real*8 colograd1d(16)

      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,4,4,
     $  ceed_gauss_lobatto,b,err)
      call ceedbasisgetcolocatedgrad(b,colograd1d)
c     TODO Not implemented in Fortran yet
c      call ceedbasisview(b,stdout,err)
      do i=0,3
        write(*,'(A,I1,A,F12.8,F12.8,F12.8,F12.8,F12.8,F12.8)')
     $ 'colograd[',i,']:',(colograd1d(j+4*i),j=1,4)
      enddo
      call ceedbasisdestroy(b,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,4,4,
     $  ceed_gauss,b,err)
      call ceedbasisgetcolocatedgrad(b,colograd1d)
c     TODO Not implemented in Fortran yet
c      call ceedbasisview(b,stdout,err)
      do i=0,3
        write(*,'(A,I1,A,F12.8,F12.8,F12.8,F12.8,F12.8,F12.8)')
     $ 'colograd[',i,']:',(colograd1d(j+4*i),j=1,4)
      enddo
      call ceedbasisdestroy(b,err)

      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
