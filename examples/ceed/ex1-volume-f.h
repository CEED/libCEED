!-----------------------------------------------------------------------
subroutine build_mass(ctx, q, j, w, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16,&
    qdata, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, ierr)
      integer*8 ctx(2)
      integer*8 fe_dim, space_dim
! j is Jacobians with shape [dim,  dim, Q]
! w is quadrature weights with shape [1, Q]
      real*8 j(1)
      real*8 w(1)
! qdata is quadrature data with shape [1, Q]
      real*8 qdata(1)
      integer q, ierr

      fe_dim = ctx(1)
      space_dim = ctx(2)

      select case (fe_dim + 10*space_dim)
        case (11)
          do i = 1, q
            qdata(i) = j(i) * w(i)
          end do

        case (22)
          do i = 1, q
            qdata(i) = (j(0*q + i)*j(3*q + i) - j(1*q + i)*j(2*q + i)) * w(i)
          end do

        case (33)
          do i = 1, q
            qdata(i) = (j(0*q + i) * (j(4*q + i)*j(8*q + i) - j(5*q + i)*j(7*q + i)) -&
                       &j(1*q + i) * (j(3*q + i)*j(8*q + i) - j(5*q + i)*j(6*q + i)) +&
                       &j(2*q + i) * (j(3*q + i)*j(7*q + i) - j(4*q + i)*j(6*q + i))) * w(i)
          end do
      end select
      ierr = 0
end

!-----------------------------------------------------------------------
subroutine apply_mass(ctx, q, u, qdata, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16,&
    v, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, ierr)
      integer*8 ctx
! u is solution variables with shape [1, Q]
! qdata is quadrature data with shape [1, Q]
      real*8 u(1)
      real*8 qdata(1)
! v is solution variables with shape [1, Q]
      real*8 v(1)
      integer q, ierr

      do i = 1, q
        v(i) = qdata(i) * u(i)
      end do
      ierr = 0
end
!-----------------------------------------------------------------------
