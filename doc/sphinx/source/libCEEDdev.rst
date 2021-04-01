Developer Notes
========================================


Shape
----------------------------------------

Backends often manipulate tensors of dimension greater than 2.  It is
awkward to pass fully-specified multi-dimensional arrays using C99 and
certain operations will flatten/reshape the tensors for computational
convenience.  We frequently use comments to document shapes using a
lexicographic ordering.  For example, the comment

.. code-block:: c

   // u has shape [dim, ncomp, Q, nelem]

means that it can be traversed as

.. code-block:: c

   for (d=0; d<dim; d++)
     for (c=0; c<ncomp; c++)
       for (q=0; q<Q; q++)
         for (e=0; e<nelem; e++)
           u[((d*ncomp + c)*Q + q)*nelem + e] = ...

This ordering is sometimes referred to as row-major or C-style.  Note
that flattening such as

.. code-block:: c

   // u has shape [dim, ncomp, Q*nelem]

and

.. code-block:: c

   // u has shape [dim*ncomp, Q, nelem]

are purely implicit -- one just indexes the same array using the
appropriate convention.


Internal Layouts
----------------------------------------

Ceed backends are free to use any **E-vector** and **Q-vector** data layout, to include never fully forming these vectors, so long as the backend passes the ``t5**`` series tests and all examples.
There are several common layouts for **L-vectors**, **E-vectors**, and **Q-vectors**, detailed below:

* **L-vector** layouts

  * **L-vectors** described by a :ref:`CeedElemRestriction` have a layout described by the ``offsets`` array and ``compstride`` parameter.
    Data for node ``i``, component ``j``, element ``k`` can be found in the **L-vector** at index ``offsets[i + k*elemsize] + j*compstride``.

  * **L-vectors** described by a strided :ref:`CeedElemRestriction` have a layout described by the ``strides`` array.
    Data for node ``i``, component ``j``, element ``k`` can be found in the **L-vector** at index ``i*strides[0] + j*strides[1] + k*strides[2]``.

* **E-vector** layouts

  * If possible, backends should use :c:func:`CeedElemRestrictionSetELayout()` to use the ``t2**`` tests.
    If the backend uses a strided **E-vector** layout, then the data for node ``i``, component ``j``, element ``k`` in the **E-vector** is given by ``i*layout[0] + j*layout[1] + k*layout[2]``.

  * Backends may choose to use a non-strided **E-vector** layout; however, the ``t2**`` tests will not function correctly in this case and the tests will need to be whitelisted for the backend to pass the test suite.

* **Q-vector** layouts

  * When the size of a :ref:`CeedQFunction` field is greater than ``1``, data for quadrature point ``i`` component ``j`` can be found in the **Q-vector** at index ``i + Q*j``.
    Backends are free to provide the quadrature points in any order.

  * When the :ref:`CeedQFunction` field has ``emode`` ``CEED_EVAL_GRAD``, data for quadrature point ``i``, component ``j``, derivative ``k`` can be found in the **Q-vector** at index ``i + Q*j + Q*size*k``.

  * Note that backend developers must take special care to ensure that the data in the **Q-vectors** for a field with ``emode`` ``CEED_EVAL_NONE`` is properly ordered when the backend uses different layouts for **E-vectors** and **Q-vectors**.


Backend Inheritance
----------------------------------------

There are three mechanisms by which a Ceed backend can inherit implementation from another Ceed backend.
These options are set in the backend initialization routine.

#. Delegation - Developers may use :c:func:`CeedSetDelegate()` to set a backend that will provide the implementation of any unimplemented Ceed objects.

#. Object delegation  - Developers may use :c:func:`CeedSetObjectDelegate()` to set a backend that will provide the implementation of a specific unimplemented Ceed object.
   Object delegation has higher precedence than delegation.

#. Operator fallback - Developers may use :c:func:`CeedSetOperatorFallbackResource()` to set a :ref:`Ceed` resource that will provide the implementation of unimplemented :ref:`CeedOperator` methods.
   A fallback :ref:`Ceed` with this resource will only be instantiated if a method is called that is not implemented by the parent :ref:`Ceed`.
   In order to use the fallback mechanism, the parent :ref:`Ceed` and fallback resource must use compatible **E-vector** and **Q-vector** layouts.


Clang-tidy
----------------------------------------

Please check your code for common issues by running

``make tidy``

which uses the ``clang-tidy`` utility included in recent releases of Clang.  This
tool is much slower than actual compilation (``make -j8`` parallelism helps).  To
run on a single file, use

``make interface/ceed.c.tidy``

for example.  All issues reported by ``make tidy`` should be fixed.


Header Files
----------------------------------------

Header inclusion for source files should follow the principal of 'include what you use' rather than relying upon transitive ``#include`` to define all symbols.

Every symbol that is used in the source file ``foo.c`` should be defined in ``foo.c``, ``foo.h``, or in a header file ``#include``d in one of these two locations.
Please check your code by running the tool ``include-what-you-use`` to see recommendations for changes to your source.
Most issues reported by ``include-what-you-use`` should be fixed; however this rule is flexible to account for differences in header file organization in external libraries.

Header files should be listed in alphabetical order, with installed headers preceding local headers and ``ceed`` headers being listed first.
