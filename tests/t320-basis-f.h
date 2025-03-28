! Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
! All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
!
! SPDX-License-Identifier: BSD-2-Clause
!
! This file is part of CEED:  http://github.com/ceed
!
!-----------------------------------------------------------------------
      subroutine buildmats(qref,qweight,interp,grad)
      integer p,q,d
      parameter(p=6)
      parameter(q=4)
      parameter(d=2)

      real*8 qref(d*q)
      real*8 qweight(q)
      real*8 interp(p*q)
      real*8 grad(d*p*q)
      real*8 x1,x2

      qref=(/2.d-1,6.d-1,1.d0/3.d0,2.d-1,2.d-1,2.d-1,1.d0/3.d0,6.d-1/)
      qweight=(/25.d0/96.d0,25.d0/96.d0,-27.d0/96.d0,25.d0/96.d0/)

      do i=0,q-1
        x1 = qref(0*q+i+1)
        x2 = qref(1*q+i+1);
!       Interp
        interp(i*P+1)=2.*(x1+x2-1.)*(x1+x2-1./2.);
        interp(i*P+2)=-4.*x1*(x1+x2-1.);
        interp(i*P+3)=2.*x1*(x1-1./2.);
        interp(i*P+4)=-4.*x2*(x1+x2-1.);
        interp(i*P+5)=4.*x1*x2;
        interp(i*P+6)=2.*x2*(x2-1./2.);
!       Grad
        grad((i+0)*P+1)=2.*(1.*(x1+x2-1./2.)+(x1+x2-1.)*1.);
        grad((i+Q)*P+1)=2.*(1.*(x1+x2-1./2.)+(x1+x2-1.)*1.);
        grad((i+0)*P+2)=-4.*(1.*(x1+x2-1.)+x1*1.);
        grad((i+Q)*P+2)=-4.*(x1*1.);
        grad((i+0)*P+3)=2.*(1.*(x1-1./2.)+x1*1.);
        grad((i+Q)*P+3)=2.*0.;
        grad((i+0)*P+4)=-4.*(x2*1.);
        grad((i+Q)*P+4)=-4.*(1.*(x1+x2-1.)+x2*1.);
        grad((i+0)*P+5)=4.*(1.*x2);
        grad((i+Q)*P+5)=4.*(x1*1.);
        grad((i+0)*P+6)=2.*0.;
        grad((i+Q)*P+6)=2.*(1.*(x2-1./2.)+x2*1.);
      enddo

      end
!-----------------------------------------------------------------------
