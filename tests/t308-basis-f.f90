!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer b
      integer u, v
      integer q, p, ncomp, dimn
      parameter(q=8,p=2,ncomp=1,dimn=3)
      integer length
      parameter(length=q**dimn)

      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)



      call ceedvectorcreate(ceed,length,u,err)
      call ceedvectorcreate(ceed,length+1,v,err)

      call ceedbasiscreatetensorh1lagrange(ceed,dimn,ncomp,p,q,ceed_gauss,b,err)
! LCOV_EXCL_START
      call ceedbasisapply(b,1,ceed_notranspose,ceed_eval_interp,u,v,err)

      call ceedbasisdestroy(b,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceeddestroy(ceed,err)

      end
! LCOV_EXCL_STOP
!-----------------------------------------------------------------------
