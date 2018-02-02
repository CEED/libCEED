c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer b,i

      integer dimn,p1d,q1d,length
      parameter(dimn   = 2)
      parameter(p1d    = 4)
      parameter(q1d    = 4)
      parameter(length = q1d**dimn)

      real*8 u(length)
      real*8 v(length)

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do i=1,length
        u(i)=1.0
      enddo

      call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,p1d,q1d,
     $  ceed_gauss_lobatto,b,err)
      call ceedbasisapply(b,ceed_notranspose,ceed_eval_interp,u,v,err)

      do i=1,length
        if (abs(v(i)-1.) > 1.D-15) then
          write(*,*) 'v(',i,'=',v(i),' not eqaul to 1.0'
        endif
      enddo

      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
