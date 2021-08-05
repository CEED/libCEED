/// @file
/// test H(div)-mixed fem for 2D quadrilateral element
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include <ceed/backend.h>

// -----------------------------------------------------------------------------
// Nodal Basis (B=[Bx;By]), xhat is in reference element [-1,1]^2
// -----------------------------------------------------------------------------
//   B = [b1,b2,b3,b4,b5,b6,b7,b8],  size(2x8),
//    local numbering is as follow (each edge has 2 dof)
//     b6     b8
//    3---------4
//  b5|         |b7
//    |         |
//  b1|         |b3
//    1---------2
//     b2     b4
// Sience nodal basis are vector, we have 16 componenets
// For example B[0] = b1_x, B[1] = b1_y, and so on
/*
static int HdivBasisQuad(CeedScalar *xhat, CeedScalar *B) {
  B[ 0] = (-xhat[0]*xhat[1] + xhat[0] + xhat[1] - 1)*0.25;
  B[ 1] = (xhat[1]*xhat[1] - 1)*0.125;
  B[ 2] = (xhat[0]*xhat[0] - 1)*0.125;
  B[ 3] = (-xhat[0]*xhat[1] + xhat[0] + xhat[1] - 1)*0.25;
  B[ 4] = (-xhat[0]*xhat[1] + xhat[0] - xhat[1] + 1)*0.25;
  B[ 5] = (xhat[1]*xhat[1] - 1)*0.125;
  B[ 6] = (-xhat[0]*xhat[0] + 1)*0.125;
  B[ 7] = (xhat[0]*xhat[1] - xhat[0] + xhat[1] - 1)*0.25;
  B[ 8] = (xhat[0]*xhat[1] + xhat[0] - xhat[1] - 1)*0.25;
  B[ 9] = (-xhat[1]*xhat[1] + 1)*0.125;
  B[10] = (xhat[0]*xhat[0] - 1)*0.125;
  B[11] = (-xhat[0]*xhat[1] - xhat[0] + xhat[1] + 1)*0.25;
  B[12] = (xhat[0]*xhat[1] + xhat[0] + xhat[1] + 1)*0.25;
  B[13] = (-xhat[1]*xhat[1] + 1)*0.125;
  B[14] = (-xhat[0]*xhat[0] + 1)*0.125;
  B[15] = (xhat[0]*xhat[1] + xhat[0] + xhat[1] + 1)*0.25;
  return 0;
}
*/

// -----------------------------------------------------------------------------
// Divergence operator; Divergence of nodal basis
// CeedScalar Dhat[8] ={0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25}
// -----------------------------------------------------------------------------

#include "t999-Hdiv2D.h"
int main(int argc, char **argv) {
  Ceed ceed;
  CeedInt P = 2, Q = 2, dim = 2;
  CeedInt nx = 1, ny = 1, num_elem = nx * ny;
  CeedInt num_nodes = (nx+1)*(ny+1), num_qpts = num_elem*Q*Q;
  CeedInt ind_x[num_elem*P*P];
  CeedScalar x[dim*num_nodes];
  CeedVector X, u, U;
  CeedElemRestriction elem_restr_x, elem_restr_mass;
  CeedBasis basis_x, basis_xc;
  CeedQFunction qf_setup;
  CeedOperator op_setup;

  CeedInit(argv[1], &ceed);

  //============= Node Coordinates, mesh [0,1]^2 square ==============
  for (CeedInt i=0; i<nx+1; i++)
    for (CeedInt j=0; j<ny+1; j++) {
      x[i+j*(nx+1)+0*num_nodes] = (CeedScalar) i / (nx);
      x[i+j*(nx+1)+1*num_nodes] = (CeedScalar) j / (ny);
    }

  CeedVectorCreate(ceed, dim*num_nodes, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  //CeedVectorView(X, "%12.8f", stdout);
  //================== Element Setup =================
  for (CeedInt i=0; i<num_elem; i++) {
    CeedInt col, row, offset;
    col = i % nx;
    row = i / nx;
    offset = col*(P-1) + row*(nx+1)*(P-1);
    for (CeedInt j=0; j<P; j++)
      for (CeedInt k=0; k<P; k++) {
        ind_x[P*(P*i+k)+j] = offset + k*(nx+1) + j;
      }
  }

  //================== Restrictions ==================
  CeedElemRestrictionCreate(ceed, num_elem, P*P, dim, num_nodes, dim*num_nodes,
                            CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restr_x);
  CeedInt strides_pb[3] = {1, Q*Q, Q *Q*16};
  CeedElemRestrictionCreateStrided(ceed, num_elem, Q*Q, 16,16*num_qpts,
                                   strides_pb, &elem_restr_mass);


  //================== H1-Basis ==================
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, P, CEED_GAUSS_LOBATTO,
                                  &basis_xc);

  //================== set-up QFunction ============
  CeedVectorCreate(ceed, num_qpts*16, &u);
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "u", 16, CEED_EVAL_NONE);

  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup);
  CeedOperatorSetField(op_setup, "dx", elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "u", elem_restr_mass, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedVectorCreate(ceed, 16, &U);
  CeedVectorSetValue(U, 0);
  CeedOperatorApply(op_setup, U, u, CEED_REQUEST_IMMEDIATE);
  CeedVectorView(u, "%12.8f", stdout);


  // Cleanup
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&U);
  CeedElemRestrictionDestroy(&elem_restr_x);
  CeedElemRestrictionDestroy(&elem_restr_mass);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_xc);
  CeedQFunctionDestroy(&qf_setup);
  CeedOperatorDestroy(&op_setup);
  CeedDestroy(&ceed);

  return 0;
}




