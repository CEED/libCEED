/// @file
/// Independent constitutive, energy-gradient and manufactured-force checks for the solids example
/// \test Independent constitutive, energy-gradient and manufactured-force checks for the solids example
// Independent constitutive checks: sigma = lambda tr(eps) I + 2 mu eps,
// W = 0.5 sigma:eps, dW/du = residual, and dR/du = Jacobian.
#include <math.h>
#include <stdio.h>
#include "../examples/solids/qfunctions/linear.h"
#include "../examples/solids/qfunctions/manufactured-force.h"
#include "../examples/solids/qfunctions/manufactured-true.h"

static int  failures = 0;
static void check(const char *name, double actual, double expected, double tol) {
  if (!isfinite(actual) || fabs(actual - expected) > tol * (1 + fabs(expected))) {
    fprintf(stderr, "%s: got %.16g, expected %.16g\n", name, actual, expected);
    failures++;
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Physics_private physics        = {.E = 2.3, .nu = 0.27};
  const double           gradients[][9] = {
      {0,    0,    0,     0.2,   0,    0,    0,    0,     0    }, // simple shear
      {0.03, 0,    0,     0,     0,    0,    0,    0,     0    }, // uniaxial strain
      {0.03, 0,    0,     0,     0.03, 0,    0,    0,     0.03 }, // dilation
      {0,    -0.2, 0,     0.2,   0,    0,    0,    0,     0    }, // infinitesimal rigid rotation
      {0.02, 0.07, -0.03, -0.01, 0.04, 0.02, 0.05, -0.06, -0.08}
  };
  for (int sample = 0; sample < 5; sample++) {
    for (int material = 0; material < 3; material++) {
      const double poisson[] = {0., 0.27, 0.49};
      physics.nu             = poisson[material];
      double qdata[10]       = {1.7, 1, 0, 0, 0, 1, 0, 0, 0, 1};
      double grad[9], residual[9], jacobian[9], energy, u[3] = {0}, diagnostic[8];
      double eps[3][3], trace = 0, expected_energy = 0;
      for (int j = 0; j < 9; j++) grad[j] = gradients[sample][j];
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) eps[j][k] = (grad[3 * k + j] + grad[3 * j + k]) / 2;
        trace += eps[j][j];
      }
      double        E        = physics.E;
      double        mu       = E / (2 * (1 + physics.nu));
      double        lambda   = E * physics.nu / ((1 + physics.nu) * (1 - 2 * physics.nu));
      const double *inputs[] = {grad, qdata};
      double       *rout[] = {residual}, *jout[] = {jacobian}, *eout[] = {&energy};
      ElasResidual_Linear(&physics, 1, inputs, rout);
      ElasJacobian_Linear(&physics, 1, inputs, jout);
      ElasEnergy_Linear(&physics, 1, inputs, eout);
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          double stress = 2 * mu * eps[j][k] + (j == k ? lambda * trace : 0);
          expected_energy += 0.5 * stress * eps[j][k];
          check("Hooke stress", residual[3 * k + j], 1.7 * stress, 1e-13);
          check("Jacobian equals linear residual", jacobian[3 * k + j], residual[3 * k + j], 1e-13);
        }
      }
      check("strain energy", energy, 1.7 * expected_energy, 1e-13);
      const double *din[]  = {u, grad, qdata};
      double       *dout[] = {diagnostic};
      ElasDiagnostic_Linear(&physics, 1, din, dout);
      check("diagnostic energy", diagnostic[7], expected_energy, 1e-13);
      for (int j = 0; j < 9; j++) {
        double  plus, minus, h = 1e-6, original = grad[j];
        double  rp[9], rm[9], direction[9] = {0}, tangent[9];
        double *ep[] = {&plus}, *em[] = {&minus}, *rpout[] = {rp}, *rmout[] = {rm};
        grad[j] = original + h;
        ElasEnergy_Linear(&physics, 1, inputs, ep);
        ElasResidual_Linear(&physics, 1, inputs, rpout);
        grad[j] = original - h;
        ElasEnergy_Linear(&physics, 1, inputs, em);
        ElasResidual_Linear(&physics, 1, inputs, rmout);
        grad[j] = original;
        check("energy gradient", (plus - minus) / (2 * h), residual[j], 1e-9);
        direction[j]         = 1;
        const double *jin[]  = {direction, qdata};
        double       *tout[] = {tangent};
        ElasJacobian_Linear(&physics, 1, jin, tout);
        for (int k = 0; k < 9; k++) check("residual derivative", (rp[k] - rm[k]) / (2 * h), tangent[k], 1e-9);
      }
    }
  }
  // Independently form -mu Laplacian(u) - (lambda + mu) grad(div(u)).
  // Each manufactured displacement component is a product of three 1D factors;
  // differentiate those factors analytically, without reusing the force formula.
  const double points[][3] = {
      {0.,  0.,  0. },
      {0.2, 0.3, 0.4},
      {0.7, 0.1, 0.8},
      {1.,  1.,  1. }
  };
  const int kind[3][3] = {
      {0, 1, 2},
      {2, 0, 1},
      {1, 2, 0}
  };  // exp, sin, cos
  const double frequencies[3] = {2., 3., 4.};
  for (int material = 0; material < 3; material++) {
    const double poisson[] = {0., 0.27, 0.49};
    physics.nu             = poisson[material];
    const double mu        = physics.E / (2 * (1 + physics.nu));
    const double lambda    = physics.E * physics.nu / ((1 + physics.nu) * (1 - 2 * physics.nu));
    for (int point = 0; point < 4; point++) {
      double x[3], force[3], truth[3], u[3], hessian[3][3][3], weight = 1.7;
      for (int axis = 0; axis < 3; axis++) x[axis] = points[point][axis];
      const double *inputs[] = {x, &weight};
      double       *fout[] = {force}, *uout[] = {truth};
      SetupMMSForce(&physics, 1, inputs, fout);
      MMSTrueSoln(NULL, 1, inputs, uout);
      for (int component = 0; component < 3; component++) {
        double factor[3][3];  // axis, derivative order
        for (int axis = 0; axis < 3; axis++) {
          const double k = frequencies[axis], t = k * x[axis];
          if (kind[component][axis] == 0) {
            factor[axis][0] = exp(t);
            factor[axis][1] = k * exp(t);
            factor[axis][2] = k * k * exp(t);
          } else if (kind[component][axis] == 1) {
            factor[axis][0] = sin(t);
            factor[axis][1] = k * cos(t);
            factor[axis][2] = -k * k * sin(t);
          } else {
            factor[axis][0] = cos(t);
            factor[axis][1] = -k * sin(t);
            factor[axis][2] = -k * k * cos(t);
          }
        }
        u[component] = factor[0][0] * factor[1][0] * factor[2][0];
        check("manufactured displacement", truth[component] * 1e8, u[component], 1e-13);
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            hessian[component][j][k] = 1.;
            for (int axis = 0; axis < 3; axis++) hessian[component][j][k] *= factor[axis][(axis == j) + (axis == k)];
          }
        }
      }
      for (int component = 0; component < 3; component++) {
        double laplacian = 0, grad_div = 0;
        for (int axis = 0; axis < 3; axis++) {
          laplacian += hessian[component][axis][axis];
          grad_div += hessian[axis][component][axis];
        }
        double expected = -weight * (mu * laplacian + (lambda + mu) * grad_div);
        check("manufactured force", force[component] * 1e8, expected, 1e-12);
      }
    }
  }
  return failures ? 1 : 0;
}
