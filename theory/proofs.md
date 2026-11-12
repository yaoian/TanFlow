# Complete proofs for the anonymous TanFlow artifact

The numbering matches Propositions 1--3 in the manuscript: minimum kinetic
closure, Sobolev error geometry, and the Schur Gaussian/point-source contrast.
All arguments below are mathematical derivations, independent of empirical
accuracy claims. The final lemma makes the position control's pushforward
precise. No external result is needed beyond finite-dimensional linear algebra,
elementary Gaussian independence, and the elementary Fourier identities
explicitly used below.

## Common notation and assumptions

Fix a conditioning observation c, two anchors \(p_a,p_b\in\mathbb R^2\), and
strictly increasing times \(t_a<\cdots<t_b\). Set \(M=b-a-1\ge1\),
\(h_i=t_{i+1}-t_i>0\), \(T=\sum_{i=a}^{b-1}h_i\), and
\(\Delta p=p_b-p_a\). There are M+1 active intervals
\(I=\{a,\ldots,b-1\}\). Inactive slots are identically zero; all statements
can therefore be proved on
\(H=\mathbb R^{2(M+1)}\), without any additional inactive dimensions.
The paper's ambient 24-slot representation is its zero extension.

Define

$$
A z=\sum_{i\in I}h_i z_i,\quad
\mathcal M_{\Delta p}=\{z\in H:Az=\Delta p\},\quad
\mathcal T_0=\ker A,\quad
\langle u,v\rangle_h=\sum_{i\in I}h_i u_i^\top v_i.
$$

For any \(q\in\mathbb R^2\), the constant field \(q/T\) has displacement q;
thus A is onto, has rank 2, and \(\dim\mathcal T_0=2M\).
The chord \(V_L=(\Delta p/T)_{i\in I}\) is feasible and
\(\mathcal M_{\Delta p}=V_L+\mathcal T_0\).
The metric matrix is \(H_h=\operatorname{diag}(h_iI_2)\succ0\).
The Euclidean transpose \(A^\top q=(h_iq)_i\) differs from the kinetic
adjoint \(A^*q=H_h^{-1}A^\top q=(q)_i\).
In particular \(AA^*=TI_2\).

## Proposition 1: Minimum-kinetic closure and tangent invariance

**Statement.** For any \(\widetilde z,w\in H\):

1. The unique minimizer of
   \(\frac12\|z-\widetilde z\|_h^2\) subject to \(Az=\Delta p\) is
   \(\Pi(\widetilde z)_i=\widetilde z_i-(A\widetilde z-\Delta p)/T\).
2. \(Pw=w-A^*Aw/T\) is the kinetic-orthogonal projection onto \(\mathcal T_0\),
   and \(\|w\|_h^2=\|Pw\|_h^2+\|Aw\|_2^2/T\).
3. Every stage and step of any explicit Runge--Kutta method with all slopes
   projected by P stays in \(\mathcal M_{\Delta p}\) when initialized there.

**Proof of weighted closure, including part 1.** More generally, let every
\(W_i\) be symmetric positive definite and minimize
\(\frac12\sum_i(z_i-\widetilde z_i)^\top W_i^{-1}(z_i-\widetilde z_i)\)
on the nonempty affine feasible set. Strict convexity gives at most one
minimizer and coercivity on this closed set gives existence. With Lagrange
multiplier \(\lambda\in\mathbb R^2\), stationarity is

$$W_i^{-1}(z_i-\widetilde z_i)+h_i\lambda=0.$$

Let \(G_W=\sum_i h_i^2 W_i\). For nonzero q,
\(q^\top G_Wq=\sum_i h_i^2q^\top W_iq>0\); hence \(G_W\) is invertible.
Enforcing the constraint gives
\(\lambda=G_W^{-1}(A\widetilde z-\Delta p)\), and consequently

$$
\Pi^{(W)}(\widetilde z)_i
=\widetilde z_i-h_iW_iG_W^{-1}(A\widetilde z-\Delta p).
$$

This point satisfies stationarity and feasibility, so it is the unique
minimizer. Alternatively, its difference from \(\widetilde z\) is orthogonal
in the \(W^{-1}\) metric to every feasible perturbation, which directly
proves minimality by a Pythagorean identity. Setting \(W_i=h_i^{-1}I_2\)
gives \(G_W=TI_2\), the kinetic objective, and the stated formula.

**Proof of part 2.** \(AP=A-AA^*A/T=0\), and \(Pw=w\) for \(Aw=0\).
Thus \(\operatorname{im}P=\ker A\) and \(P^2=P\). Also

$$
\langle Pu,v\rangle_h
=\langle u,v\rangle_h-\frac{(Au)^\top Av}{T}
=\langle u,Pv\rangle_h.
$$

So P is self-adjoint in the kinetic metric. Its complementary image is
\(\operatorname{im}A^*\), the two-dimensional constant-velocity space.
For \(u\in\ker A\), \(\langle u,A^*q\rangle_h=(Au)^\top q=0\).
The decomposition \(w=Pw+A^*Aw/T\) is therefore orthogonal and unique.
Finally \(\|A^*Aw/T\|_h^2=T\|Aw/T\|_2^2=\|Aw\|_2^2/T\), proving the norm
identity. It also follows that \(\Pi(\widetilde z)=P\widetilde z+V_L\).

**Proof of part 3.** Consider an arbitrary explicit r-stage step with size
\(\eta\), stage coefficients \(a_{jq}\), final coefficients \(b_j\), and
stage times \(\tau_j\):

$$
Z_j=z^{(k)}+\eta\sum_{q<j}a_{jq}k_q,\quad
k_j=P f_\theta(Z_j,\tau_j,c),\quad
z^{(k+1)}=z^{(k)}+\eta\sum_j b_jk_j.
$$

Every finite raw slope satisfies \(Ak_j=0\), regardless of network weights.
If \(Az^{(k)}=\Delta p\), applying A to each stage and the update yields
\(AZ_j=Az^{(k+1)}=\Delta p\). Induction over steps from the assumed
initialization proves the result; stage induction ensures all field calls
are feasible. No consistency or positivity assumption on the RK
coefficients is needed for this algebraic invariant. Heun is the special
case \(k_1=Pf(z,\tau)\), \(k_2=Pf(z+\eta k_1,\tau+\eta)\), and
\(z^+=z+\eta(k_1+k_2)/2\). Floating-point roundoff is outside this
exact-arithmetic statement. QED.

## Proposition 2: Sobolev envelope and bilateral handoffs

**Statement.** Let \(z,v^*\in\mathcal M_{\Delta p}\), \(e=z-v^*\), and let
\(e(u)=e_i\) on each interval \([t_i,t_{i+1})\). Define
\(E(t)=\int_{t_a}^t e(u)\,du\), and write \(s=t-t_a\).
Then \(E\in H^1_0([t_a,t_b];\mathbb R^2)\), with
\(\|E'\|_{L^2}=\|e\|_h\), and

$$
\|E(t)\|_2\le\sqrt{\frac{s(T-s)}T}\|e\|_h,\qquad
\|E\|_{L^2}\le\frac T\pi\|e\|_h.
$$

For \(w_j=(t_j-t_a)(t_b-t_j)/T\), the discrete metrics obey

$$
\mathrm{PointMSE}\le\frac1M\sum_{j=a+1}^{b-1}w_j\|e\|_h^2,\qquad
\mathrm{ADE}\le\frac1M\sum_{j=a+1}^{b-1}\sqrt{w_j}\|e\|_h.
$$

The exact boundary identities are
\(E_{a+1}=h_ae_a\) and \(E_{b-1}=-h_{b-1}e_{b-1}\). Thus the corresponding
error norms are \(h_a\|e_a\|_2\) and
\(\mathrm{FDE}_{\rm miss}=h_{b-1}\|e_{b-1}\|_2\), bounded respectively by
\(\sqrt{h_a(T-h_a)/T}\|e\|_h\) and
\(\sqrt{h_{b-1}(T-h_{b-1})/T}\|e\|_h\).

**Proof.** The step function e is square-integrable. Its integral E is
absolutely continuous, piecewise affine, and has weak derivative e.
Moreover \(E(t_a)=0\) and \(E(t_b)=Ae=Az-Av^*=0\), so its endpoint traces
vanish and it belongs to \(H^1_0\).
Direct integration gives \(\int\|e(u)\|_2^2du=\sum_i h_i\|e_i\|_2^2\).

Since the total integral of e is zero, for any s in [0,T],

$$
E(t_a+s)=\int_0^T g_s(u)e(t_a+u)\,du,\qquad
g_s(u)=\mathbf1_{[0,s]}(u)-s/T.
$$

The vector-valued Cauchy--Schwarz inequality gives
\(\|E(t_a+s)\|_2^2\le\|g_s\|_{L^2}^2\|e\|_h^2\). Explicitly,

$$
\|g_s\|_{L^2}^2=s(1-s/T)^2+(T-s)(s/T)^2=s(T-s)/T.
$$

This proves the envelope including both endpoints. Evaluating it at every
missing timestamp and averaging its squared or unsquared form proves the
two discrete inequalities. No uniform-time assumption was used.

For completeness, the constant in the integrated inequality follows
componentwise from Fourier expansion. For a scalar \(f\in H^1_0(0,T)\),
let \(c_n=\int_0^T f(s)\sqrt{2/T}\sin(n\pi s/T)\,ds\).
Parseval gives \(\|f\|_{L^2}^2=\sum_{n\ge1}|c_n|^2\).
Integration by parts in the weak sense, using the zero endpoint traces,
shows that the coefficient of \(f'\) against
\(\sqrt{2/T}\cos(n\pi s/T)\) is \((n\pi/T)c_n\).
The constant cosine coefficient is
\(T^{-1/2}\int_0^T f'=0\). The cosine basis is complete, so
\(\|f'\|_{L^2}^2=\sum_{n\ge1}(n\pi/T)^2|c_n|^2\).
These equalities may first be checked on smooth functions with zero traces
and extended by their density in \(H^1_0\). Hence
\(\|f\|_{L^2}^2\le(T/\pi)^2\|f'\|_{L^2}^2\). Summing the two component
inequalities proves the vector bound. The constant is sharp in the full
Sobolev space, as witnessed by \(E(t_a+s)=q\sin(\pi s/T)\);
equality need not occur in a prescribed finite piecewise-linear grid.

Finally, prefix integration gives \(E_{a+1}-E_a=h_ae_a\), whereas the last
interval gives \(E_b-E_{b-1}=h_{b-1}e_{b-1}\). Both anchor errors vanish.
Taking norms and evaluating the envelope at s equal to \(h_a\) and
\(T-h_{b-1}\) proves all handoff statements. QED.

The pointwise constant is also sharp in \(H^1_0\): for fixed interior s
choose \(e(t_a+u)=g_s(u)q\). It has zero integral and attains equality.
If s is a grid timestamp this example is also a valid intervalwise-constant
velocity error. These bounds relate errors; they do not imply that minimum
velocity loss and minimum discrete ADE select the same predictor.

## Proposition 3: Schur Gaussian and the point-source shortcut

**Statement and probabilistic scope.** Condition on a fixed c. Let
\(\mu=V_K(c)\in\mathcal M_{\Delta p}\) and let the target \(Z_1\) be feasible
almost surely with finite first moment. Let
\(\Sigma=\operatorname{diag}(\Sigma_i)\succ0\) on H and draw
\(\xi\sim\mathcal N(0,\Sigma)\), independently of \(Z_1\) given c.
Define \(G=A\Sigma A^\top\), \(K=\Sigma A^\top G^{-1}\), and \(Q=I-KA\).
Then

$$
Z_0=\Pi^{(\Sigma)}(\mu+\xi)=\mu+Q\xi,\qquad
\Sigma_{\rm tan}=\Sigma-\Sigma A^\top G^{-1}A\Sigma
$$

is an exact conditional Gaussian sample on \(\mathcal M_{\Delta p}\).
Its covariance image is \(\mathcal T_0\) and its rank is 2M.
For \(Z_\tau=(1-\tau)Z_0+\tau Z_1\), the states are feasible and
\(\bar u_\tau(Z_\tau,c)=\mathbb E[Z_1-Z_0\mid Z_\tau,c]\) is tangent,
almost surely for each \(\tau\in[0,1]\).

If instead \(Z_0=\mu\) is deterministic (S0), then for \(\tau>0\) the
canonical conditional field is
\(\bar u_\tau(z,c)=(z-\mu)/\tau\) on the affine manifold. Its derivative
along \(\mathcal T_0\) is \(I_{\mathcal T_0}/\tau\). At \(\tau=0\), only the
value at \(\mu\) is identified:
\(\bar u_0(\mu,c)=\mathbb E[Z_1\mid c]-\mu\).

**Proof of the Gaussian assertion.** As A has full row rank and Sigma is
positive definite, \(q^\top Gq=(A^\top q)^\top\Sigma(A^\top q)>0\) for
nonzero q; hence G is invertible. Thus \(AK=I_2\), \(AQ=0\), \(Q^2=Q\),
and Q acts as the identity on \(\ker A\). The weighted formula of
Proposition 1 with \(W_i=\Sigma_i\), together with \(A\mu=\Delta p\),
gives \(Z_0=\mu+Q\xi\).

Let \(Y=A\xi\) and \(\eta=Q\xi\). The pair is jointly Gaussian, including
the possibility of a singular marginal for eta, and

$$
\operatorname{Cov}(\eta,Y)=Q\Sigma A^\top
=\Sigma A^\top-KG=0.
$$

Joint Gaussian vectors with zero cross-covariance are independent: their
joint characteristic function factors because its quadratic covariance
form has no cross term. Also \(\xi=\eta+KY\). Consequently a regular
conditional law of \(\xi\) given \(Y=y\) is the law of \(\eta+Ky\).
This explicitly defines the continuous Gaussian conditional version even
at \(y=0\), an event of probability zero. It is not conditioning by a
ratio of probabilities of a zero-probability event.
Conditioning on \(A(\mu+\xi)=\Delta p\) is \(Y=0\), so the resulting law
is exactly that of \(\mu+\eta\).

Expanding its covariance and using \(G=A\Sigma A^\top\) gives

$$
Q\Sigma Q^\top
=\Sigma-KA\Sigma-\Sigma A^\top K^\top+KGK^\top
=\Sigma-\Sigma A^\top G^{-1}A\Sigma=\Sigma_{\rm tan}.
$$

Since Q fixes \(\ker A\) and maps into it,
\(\operatorname{im}Q=\ker A\) and \(\operatorname{rank}Q=2M\).
The invertible square root \(\Sigma^{1/2}\) implies
\(\operatorname{rank}(Q\Sigma^{1/2})=2M\). For any matrix B,
\(\ker(BB^\top)=\ker B^\top\), so
\(\operatorname{rank}(BB^\top)=\operatorname{rank}B\).
Taking \(B=Q\Sigma^{1/2}\) proves rank \(2M\).
Since \(A\Sigma_{\rm tan}=0\), its image is contained in \(\ker A\);
equal dimensions give equality. A Gaussian with this covariance has full
relative support \(\mu+\ker A=\mathcal M_{\Delta p}\).

On the probability-one event where both endpoints are feasible,
\(AZ_\tau=(1-\tau)\Delta p+\tau\Delta p=\Delta p\) simultaneously for all
tau. The finite first moments make the conditional expectations well-defined.
Linearity and boundedness of A in finite dimensions give, for each tau,

$$
A\bar u_\tau(Z_\tau,c)
=\mathbb E[A(Z_1-Z_0)\mid Z_\tau,c]=0.
$$

This is a statement about a conditional-expectation version almost surely,
not an assertion about arbitrary off-support neural extrapolations.

At tau zero, conditional independence gives
\(\mathbb E[Z_1-Z_0\mid Z_0,c]=\mathbb E[Z_1\mid c]-Z_0\).
For a full-rank tangent Gaussian its canonical version is affine on the
entire feasible manifold with tangent Jacobian \(-I_{\mathcal T_0}\).
This is an exact statement at zero, not a uniform derivative bound on any
neighborhood of zero in time and not a regularity theorem for a fitted network.

**Proof of the S0 assertion.** For fixed \(\tau>0\),
\(Z_\tau=\mu+\tau(Z_1-\mu)\), with inverse
\(Z_1=\mu+(Z_\tau-\mu)/\tau\). Thus \(Z_1-\mu\) is measurable with
respect to \(Z_\tau,c\); taking conditional expectation leaves it unchanged.
This proves \(\bar u_\tau(Z_\tau,c)=(Z_\tau-\mu)/\tau\) almost surely,
without requiring a density. The displayed affine formula defines a
canonical smooth version on all of \(\mathcal M_{\Delta p}\). In any
orthonormal tangent coordinates \(z=\mu+Ux\), its coordinate field is
\(x/\tau\), whose Jacobian is \(I_{2M}/\tau\) and operator norm \(1/\tau\).
The same scalar-identity norm holds with the kinetic tangent metric.

If the target has a density with support of nonempty relative interior, this
derivative describes the canonical field on that interior. Conditional
expectations are identified only almost surely; an arbitrary version modified
on a null set need not be differentiable. If a continuous version is required
on the support, equality almost surely determines it there, since every open
neighborhood of a support point has positive probability. At tau zero,
conditioning on the deterministic source adds no information beyond c, giving
the stated conditional mean. QED.

**Limit of the conclusion at zero.** All rays
\(z(\tau)=\mu+\tau v\) solve the canonical field equation for \(\tau>0\).
They share the limiting position mu. If differentiability and the mean-field
ODE are imposed at exactly zero, their initial derivative must equal the
conditional mean displacement. The singular equation for positive tau
alone does not establish existence, uniqueness, or a learned solver's
behavior at zero. Likewise full-rank S1/S2 support is not, by itself, a
global smoothness or numerical-stability guarantee. The source is randomized
for training; reported inference starts at the center.

## Lemma: Prefix-Integration Affine Bijection

**Statement.** Fix the same anchors and positive durations. The map

$$
D\colon\mathcal M_{\Delta p}\longrightarrow\mathbb R^{2M},\qquad
(Dz)_j=p_a+\sum_{i=a}^{j-1}h_i z_i,\quad a<j<b,
$$

is an affine bijection. Its linear part restricted to \(\mathcal T_0\)
is an isomorphism. Thus every feasible velocity law has an exact position
pushforward, and no representational degrees of freedom are lost.

**Proof by an explicit inverse.** Given arbitrary interior positions
\(x=(x_{a+1},\ldots,x_{b-1})\in\mathbb R^{2M}\), append
\(x_a=p_a,x_b=p_b\), and define

$$
(D^{-1}x)_i=(x_{i+1}-x_i)/h_i,\quad i=a,\ldots,b-1.
$$

This is well-defined since every \(h_i>0\). Telescoping gives
\(A(D^{-1}x)=\sum_i(x_{i+1}-x_i)=p_b-p_a=\Delta p\), so its image is
feasible. Prefix summation telescopes to
\((DD^{-1}x)_j=p_a+x_j-x_a=x_j\).
Conversely, for feasible z, append the anchors to Dz. Their consecutive
differences are \(h_iz_i\), including the last interval because \(Az=\Delta p\).
Therefore \(D^{-1}Dz=z\). Both maps are affine, proving the bijection.

Write \(Dz=d+Lz\), where d is the repeated left anchor and L is the prefix
matrix. For a tangent perturbation e, \(Le\) consists of interior errors.
Its inverse appends zero endpoint errors and takes successive differences
divided by \(h_i\); telescoping makes that inverse tangent. Hence
\(L|_{\mathcal T_0}\) is an isomorphism onto \(\mathbb R^{2M}\).
In particular \(D((1-\tau)z_0+\tau z_1)=(1-\tau)Dz_0+\tau Dz_1\).
If \(Z_0\sim\mathcal N(\mu,\Sigma_{\rm tan})\), then

$$
DZ_0\sim\mathcal N(D\mu,L\Sigma_{\rm tan}L^\top).
$$

This position covariance is positive definite: \(\Sigma_{\rm tan}\) has
image \(\mathcal T_0\), and \(L|_{\mathcal T_0}\) is onto, so
\(L\Sigma_{\rm tan}^{1/2}\) has rank \(2M\); its product with its transpose
therefore has full rank. The claim applies to interior positions; a packed
24-slot representation with fixed visible positions has additional
deterministic coordinates and is not full rank in all packed coordinates.
QED.

**Notation and induced metric in the concise manuscript.** The manuscript
calls the full affine map above \(\Phi\), and reserves \(D_{\rm red}\)
(written D there) for its square matrix after eliminating the last velocity.
Writing \(w=(z_a,\ldots,z_{b-2})\), its formula is
\(\Phi(z)=\mathbf1_M\otimes p_a+D_{\rm red}w\), where
\(D_{\rm red}=(L_{\rm low}\operatorname{diag}(h_a,\ldots,h_{b-2}))\otimes I_2\)
and \(L_{\rm low}\) is the unit lower-triangular all-ones matrix.
Its determinant is \(\prod_{i=a}^{b-2}h_i^2>0\), agreeing with the inverse
already proved. Its Gram block at indices i,k is
\(h_i h_k [M-\max(i-a,k-a)]I_2\): exactly that many prefix sums contain both
intervals. Thus the sum of squared interior position errors is
\(e_{a:b-2}^\top D_{\rm red}^\top D_{\rm red}e_{a:b-2}\).
The omitted exit velocity is fixed by
\(h_{b-1}e_{b-1}=-\sum_{i=a}^{b-2}h_i e_i\), not an independent zero-cost
variable. For nonuniform durations the diagonal weights also contain
\(h_i^2\), so an ordering of their magnitudes by interval index alone
requires additional assumptions.

**Consequence for comparison.** Under an exactly conjugated field
\(g(x,\tau)=Lf(D^{-1}x,\tau)\), affine changes of variables commute with
explicit RK stages. Position-FM, however, trains a separate neural field
and uses a position loss; conjugacy is not imposed. For example its squared
position error is \(\|Le\|_2^2\), whereas active velocity loss is
proportional to \(\|e\|_2^2\), and kinetic loss is \(\|e\|_h^2\).
These generally differ. The matched pushforward source and parameter count
therefore control representation and capacity, without implying identical
objectives, optimization, learned fields, or empirical errors.
