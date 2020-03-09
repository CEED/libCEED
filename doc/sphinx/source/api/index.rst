**************************************
API Documentation
**************************************

This section contains the code documentation. The subsections represent
the different API objects, typedefs, and enumerations.


Public API
======================================

These objects and functions are intended to be used by general users of libCEED
and can generally be found in `ceed.h`.


.. only:: html

   .. mermaid::

      graph TD
      U -->|wrap data| V
      U(High-level user code) -->|apply| O
      subgraph Ceed
      O(CeedOperator) --> E(CeedElemRestriction)
      O --> B(CeedBasis)
      O --> Q(CeedQFunction)
      B --> V(CeedVector)
      O --> V
      E --> V
      Q --> V
      end
      Q --> UQ(User's physics)

.. toctree::
   :maxdepth: 4

   Ceed
   CeedVector
   CeedElemRestriction
   CeedBasis
   CeedQFunction
   CeedOperator


Backend API
======================================

These functions are intended to be used by backend developers of libCEED and can
generally be found in `ceed-backend.h`.

.. toctree::

   backend/Ceed
   backend/CeedVector
   backend/CeedElemRestriction
   backend/CeedBasis
   backend/CeedQFunction
   backend/CeedOperator


Internal Functions
======================================

These functions are intended to be used by library developers of libCEED and can
generally be found in `ceed-impl.h`.

.. toctree::

   internal/Ceed
   internal/CeedVector
   internal/CeedElemRestriction
   internal/CeedBasis
   internal/CeedQFunction
   internal/CeedOperator
