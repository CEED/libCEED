# libCEED Developer Notes

## Shape

Backends often manipulate tensors of dimension greater than 2.  It is
awkward to pass fully-specified multi-dimensional arrays using C99 and
certain operations will flatten/reshape the tensors for computational
convenience.  We frequently use comments to document shapes using a
lexicographic ordering.  For example, the comment

    // u has shape [dim, ncomp, Q, nelem]

means that it can be traversed as

    for (d=0; d<dim; d++)
      for (c=0; c<ncomp; c++)
        for (q=0; q<Q; q++)
          for (e=0; e<nelem; e++)
            u[((d*ncomp + c)*Q + q)*nelem + e] = ...

This ordering is sometimes referred to as row-major or C-style.  Note
that flattening such as

    // u has shape [dim, ncomp, Q*nelem]

and

    // u has shape [dim*ncomp, Q, nelem]

are purely implicit -- one just indexes the same array using the
appropriate convention.
