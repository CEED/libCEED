!-----------------------------------------------------------------------
subroutine build_diff(ctx, q, dx, w, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16,&
    qdata, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, ierr)
      integer*8 ctx(2)
      integer*8 dim, space_dim
! dx is Jacobians with shape [dim, dim, Q]
! w is quadrature weights with shape [1, Q]
      real*8 dx(1)
      real*8 w(1)
! qdata is quadrature data with shape [dim*(dim+1)/2, Q]
      real*8 qdata(1)
      integer q, ierr

      real*8 J00, J10, J01, J11, det, qw
      real*8 A(3, 3)
      integer i, j, k

      dim       = ctx(1)
      space_dim = ctx(2)

      select case (dim + 10*space_dim)

        case (11)
!         q_data[0][i] = w[i] / J[0][0][i]
          do i = 1, q
            qdata(i) = w(i) / dx(i)
          end do

        case (22)
!         J layout (column-major, each block of Q):
!           dx(0*q+i) = J[0][0], dx(1*q+i) = J[0][1]
!           dx(2*q+i) = J[1][0], dx(3*q+i) = J[1][1]
!         qdata layout [3, Q] Voigt:
!           qdata(0*q+i) = D00, qdata(1*q+i) = D11, qdata(2*q+i) = D01
          do i = 1, q
            J00 = dx(0*q + i)
            J10 = dx(1*q + i)
            J01 = dx(2*q + i)
            J11 = dx(3*q + i)
            det = J00*J11 - J10*J01
            qw  = w(i) / det
            qdata(0*q + i) = qw * (J01*J01 + J11*J11)
            qdata(1*q + i) = qw * (J00*J00 + J10*J10)
            qdata(2*q + i) = -qw * (J00*J01 + J10*J11)
          end do

        case (33)
!         J layout (column-major, each block of Q):
!           dx(col*3*q + row*q + i) = J[col][row][i]
!         qdata layout [6, Q] Voigt:
!           0=D00, 1=D11, 2=D22, 3=D12, 4=D02, 5=D01
          do i = 1, q
!           Build adjugate A[k][j] = cofactor of J[j][k]
            do j = 0, 2
              do k = 0, 2
                A(k+1, j+1) = &
                  dx(mod(k+1,3)*3*q + mod(j+1,3)*q + i) * dx(mod(k+2,3)*3*q + mod(j+2,3)*q + i) - &
                  dx(mod(k+2,3)*3*q + mod(j+1,3)*q + i) * dx(mod(k+1,3)*3*q + mod(j+2,3)*q + i)
              end do
            end do
!           det(J) = J[0][0]*A[0][0] + J[0][1]*A[1][0] + J[0][2]*A[2][0]
            det = dx(0*3*q + 0*q + i) * A(1,1) + &
                  dx(0*3*q + 1*q + i) * A(2,1) + &
                  dx(0*3*q + 2*q + i) * A(3,1)
            qw = w(i) / det
!           Store symmetric part in Voigt convention:
!           0 5 4
!           5 1 3
!           4 3 2
            qdata(0*q + i) = qw * (A(1,1)*A(1,1) + A(1,2)*A(1,2) + A(1,3)*A(1,3))
            qdata(1*q + i) = qw * (A(2,1)*A(2,1) + A(2,2)*A(2,2) + A(2,3)*A(2,3))
            qdata(2*q + i) = qw * (A(3,1)*A(3,1) + A(3,2)*A(3,2) + A(3,3)*A(3,3))
            qdata(3*q + i) = qw * (A(2,1)*A(3,1) + A(2,2)*A(3,2) + A(2,3)*A(3,3))
            qdata(4*q + i) = qw * (A(1,1)*A(3,1) + A(1,2)*A(3,2) + A(1,3)*A(3,3))
            qdata(5*q + i) = qw * (A(1,1)*A(2,1) + A(1,2)*A(2,2) + A(1,3)*A(2,3))
          end do

      end select
      ierr = 0
end

!-----------------------------------------------------------------------
subroutine apply_diff(ctx, q, du, qdata, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16,&
    dv, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, ierr)
      integer*8 ctx(2)
      integer*8 dim
! du is solution gradient with shape [dim, 1, Q]
! qdata is quadrature data with shape [dim*(dim+1)/2, Q]
      real*8 du(1)
      real*8 qdata(1)
! dv is output gradient with shape [dim, 1, Q]
      real*8 dv(1)
      integer q, ierr

      integer i, j
      real*8 dXdxdXdx_T(3, 3)

      dim = ctx(1)

      select case (dim)

        case (1)
!         dv[i] = du[i] * qdata[0][i]
          do i = 1, q
            dv(i) = du(i) * qdata(i)
          end do

        case (2)
!         du layout [2, Q]: du(0*q+i), du(1*q+i)
!         qdata Voigt [3, Q]:
!           0 2
!           2 1
!         dv[j][i] = sum_k( du[k][i] * dXdxdXdx_T[k][j] )
          do i = 1, q
            dXdxdXdx_T(1,1) = qdata(0*q + i)
            dXdxdXdx_T(1,2) = qdata(2*q + i)
            dXdxdXdx_T(2,1) = qdata(2*q + i)
            dXdxdXdx_T(2,2) = qdata(1*q + i)
            do j = 1, 2
              dv((j-1)*q + i) = du(0*q + i) * dXdxdXdx_T(1, j) + &
                                 du(1*q + i) * dXdxdXdx_T(2, j)
            end do
          end do

        case (3)
!         du layout [3, Q]: du(0*q+i), du(1*q+i), du(2*q+i)
!         qdata Voigt [6, Q]:
!           0 5 4
!           5 1 3
!           4 3 2
!         dv[j][i] = sum_k( du[k][i] * dXdxdXdx_T[k][j] )
          do i = 1, q
            dXdxdXdx_T(1,1) = qdata(0*q + i)
            dXdxdXdx_T(1,2) = qdata(5*q + i)
            dXdxdXdx_T(1,3) = qdata(4*q + i)
            dXdxdXdx_T(2,1) = qdata(5*q + i)
            dXdxdXdx_T(2,2) = qdata(1*q + i)
            dXdxdXdx_T(2,3) = qdata(3*q + i)
            dXdxdXdx_T(3,1) = qdata(4*q + i)
            dXdxdXdx_T(3,2) = qdata(3*q + i)
            dXdxdXdx_T(3,3) = qdata(2*q + i)
            do j = 1, 3
              dv((j-1)*q + i) = du(0*q + i) * dXdxdXdx_T(1, j) + &
                                 du(1*q + i) * dXdxdXdx_T(2, j) + &
                                 du(2*q + i) * dXdxdXdx_T(3, j)
            end do
          end do

      end select
      ierr = 0
end
!-----------------------------------------------------------------------