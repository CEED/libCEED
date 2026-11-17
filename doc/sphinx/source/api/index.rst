**************************************
API Documentation
**************************************

This section contains the code documentation. The subsections represent
the different API objects, typedefs, and enumerations.


Public API
======================================

These objects and functions are intended to be used by general users of libCEED and can generally be found in `ceed.h`.


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
   :maxdepth: 2

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
   :maxdepth: 2

   Ceed <backend/Ceed>
   CeedVector <backend/CeedVector>
   CeedElemRestriction <backend/CeedElemRestriction>
   CeedBasis <backend/CeedBasis>
   CeedQFunction <backend/CeedQFunction>
   CeedOperator <backend/CeedOperator>


Internal Functions
======================================

These functions are intended to be used by library developers of libCEED and can
generally be found in `ceed-impl.h`.

.. toctree::
   :maxdepth: 2

   Ceed <internal/Ceed>
   CeedVector <internal/CeedVector>
   CeedElemRestriction <internal/CeedElemRestriction>
   CeedBasis <internal/CeedBasis>
   CeedQFunction <internal/CeedQFunction>
   CeedOperator <internal/CeedOperator>
