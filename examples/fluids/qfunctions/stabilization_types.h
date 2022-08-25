#ifndef stabilization_types_h
#define stabilization_types_h

typedef enum {
  STAB_NONE = 0,
  STAB_SU   = 1,  // Streamline Upwind
  STAB_SUPG = 2,  // Streamline Upwind Petrov-Galerkin
} StabilizationType;

#endif  // stabilization_types_h
