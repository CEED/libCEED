c
c Dummy parameters for CEED Fortran 77 stubs
c

c
c CeedMemType
c

      integer ceed_mem_host
      parameter(ceed_mem_host   = 0)

      integer ceed_mem_device
      parameter(ceed_mem_device = 1)

c
c CeedCopyMode
c

      integer ceed_copy_values
      parameter(ceed_copy_values = 0)

      integer ceed_use_pointer
      parameter(ceed_use_pointer = 1)

      integer ceed_own_pointer
      parameter(ceed_own_pointer = 2)

c
c CeedRequest related
c

      integer ceed_request_immediate
      parameter(ceed_request_immediate = -1)

      integer ceed_request_ordered
      parameter(ceed_request_ordered   = -2)

c
c Null
c

      integer ceed_null
      parameter(ceed_null = -3)

c
c CeedTransposeMode
c

      integer ceed_notranspose
      parameter(ceed_notranspose = 0)

      integer ceed_transpose
      parameter(ceed_transpose   = 1)

c
c CeedEvalMode
c

      integer ceed_eval_none
      parameter(ceed_eval_none   = 0 )

      integer ceed_eval_interp
      parameter(ceed_eval_interp = 1 )

      integer ceed_eval_grad
      parameter(ceed_eval_grad   = 2 )

      integer ceed_eval_div
      parameter(ceed_eval_div    = 4 )

      integer ceed_eval_curl
      parameter(ceed_eval_curl   = 8 )

      integer ceed_eval_weight
      parameter(ceed_eval_weight = 16)

c
c CeedQualMode
c

      integer ceed_gauss
      parameter(ceed_gauss         = 0)

      integer ceed_gauss_lobatto
      parameter(ceed_gauss_lobatto = 1)

c
c CeedElemTopology
c

      integer ceed_line
      parameter(ceed_line        = X'10000' )

      integer ceed_triangle
      parameter(ceed_triangle    = X'20001' )

      integer ceed_quad
      parameter(ceed_quad        = X'20002' )

      integer ceed_tet
      parameter(ceed_tet         = X'30003' )

      integer ceed_pryamid
      parameter(ceed_pryamid     = X'30004' )

      integer ceed_prism
      parameter(ceed_prism       = X'30005' )

      integer ceed_hex
      parameter(ceed_hex         = X'30006' )

c
c OperatorFieldConstants
c

      integer ceed_basis_collocated
      parameter(ceed_basis_collocated     = -1)

      integer ceed_vector_active
      parameter(ceed_vector_active        = -1)

      integer ceed_vector_none
      parameter(ceed_vector_none          = -2)
