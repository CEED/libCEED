(bps)=

# CEED Bakeoff Problems

```{include} ./README.md
:start-after: bps-inclusion-marker
:end-before: bps-exclusion-marker
```

(mass-operator)=

## Mass Operator

The Mass Operator used in BP1 and BP2 is defined via the $L^2$ projection
problem, posed as a weak form on a Hilbert space $V^p \subset H^1$, i.e.,
find $u \in V^p$ such that for all $v \in V^p$

$$
\langle v,u \rangle = \langle v,f \rangle ,
$$ (eq-general-weak-form)

where $\langle v,u\rangle$ and $\langle v,f\rangle$ express the continuous
bilinear and linear forms, respectively, defined on $V^p$, and, for sufficiently
regular $u$, $v$, and $f$, we have:

$$
\begin{aligned} \langle v,u \rangle &:= \int_{\Omega} \, v \, u \, dV ,\\ \langle v,f \rangle &:= \int_{\Omega} \, v \, f \, dV . \end{aligned}
$$

Following the standard finite/spectral element approach, we formally
expand all functions in terms of basis functions, such as

$$
\begin{aligned}
u(\bm x) &= \sum_{j=1}^n u_j \, \phi_j(\bm x) ,\\
v(\bm x) &= \sum_{i=1}^n v_i \, \phi_i(\bm x) .
\end{aligned}
$$ (eq-nodal-values)

The coefficients $\{u_j\}$ and $\{v_i\}$ are the nodal values of $u$
and $v$, respectively. Inserting the expressions {math:numref}`eq-nodal-values`
into {math:numref}`eq-general-weak-form`, we obtain the inner-products

$$
\langle v,u \rangle = \bm v^T M \bm u , \qquad  \langle v,f\rangle =  \bm v^T \bm b \,.
$$ (eq-inner-prods)

Here, we have introduced the mass matrix, $M$, and the right-hand side,
$\bm b$,

$$
M_{ij} :=  (\phi_i,\phi_j), \;\; \qquad b_{i} :=  \langle \phi_i, f \rangle,
$$

each defined for index sets $i,j \; \in \; \{1,\dots,n\}$.

(laplace-operator)=

## Laplace's Operator

The Laplace's operator used in BP3-BP6 is defined via the following variational
formulation, i.e., find $u \in V^p$ such that for all $v \in V^p$

$$
a(v,u) = \langle v,f \rangle , \,
$$

where now $a (v,u)$ expresses the continuous bilinear form defined on
$V^p$ for sufficiently regular $u$, $v$, and $f$, that is:

$$
\begin{aligned} a(v,u) &:= \int_{\Omega}\nabla v \, \cdot \, \nabla u \, dV ,\\ \langle v,f \rangle &:= \int_{\Omega} \, v \, f \, dV . \end{aligned}
$$

After substituting the same formulations provided in {math:numref}`eq-nodal-values`,
we obtain

$$
a(v,u) = \bm v^T K \bm u ,
$$

in which we have introduced the stiffness (diffusion) matrix, $K$, defined as

$$
K_{ij} = a(\phi_i,\phi_j),
$$

for index sets $i,j \; \in \; \{1,\dots,n\}$.
