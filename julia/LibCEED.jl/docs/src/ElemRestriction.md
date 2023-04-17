# ElemRestriction

```@docs
ElemRestriction
ElemRestrictionNone
create_elem_restriction
create_elem_restriction
create_elem_restriction_oriented
create_elem_restriction_curl_oriented
create_elem_restriction_strided
apply!(r::ElemRestriction, u::CeedVector, ru::CeedVector; tmode=NOTRANSPOSE, request=RequestImmediate())
apply(r::ElemRestriction, u::AbstractVector; tmode=NOTRANSPOSE)
create_evector
create_lvector
create_vectors
getcompstride
getnumelements
getelementsize
getlvectorsize
getnumcomponents(r::ElemRestriction)
getmultiplicity!
getmultiplicity
```
