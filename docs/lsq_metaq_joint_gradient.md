# Joint LSQ-METaQ gradient

## Objective

For one quantized tensor, let the signed integer LSQ codebook be `q` and let
the current step size be `s > 0`. The non-zero METaQ bucket values are

\[
v_b(s)=s q_b.
\]

For fixed network weights `w`, define the perspective METaQ value function

\[
\phi(w,s)=\min_{x,y,c}
T_1\sum_i\frac{w_i^2}{y_i}
+T_2\sum_b c_b\log c_b
+T_3\sum_i y_i
\]

subject to

\[
w_i-\sum_b s q_b x_{i,b}=0,\qquad
\sum_bx_{i,b}=y_i,\qquad
c_b=\sum_i x_{i,b},\qquad 0\le y_i\le1.
\]

The zero output of the signed int4 quantizer is represented by the missing
mass `z_i = 1-y_i`, rather than by a second zero-valued METaQ bucket. Thus an
int4 tensor has 15 non-zero bucket values and one zero symbol.

## Envelope derivatives

Use the Lagrangian sign convention

\[
\mathcal L=F(x,y,c;w)+\sum_i\mu_i
\left(w_i-\sum_b s q_bx_{i,b}\right)+\cdots.
\]

At a regular primal-dual solution, the envelope theorem gives

\[
\frac{\partial\phi}{\partial w_i}
=\mu_i+2T_1\frac{w_i}{y_i^*}
\]

and

\[
\boxed{
\frac{\partial\phi}{\partial s}
=-\sum_i\mu_i\sum_bq_bx_{i,b}^*
=-\frac1s\sum_i\mu_iw_i.}
\]

The second equality uses primal feasibility. Importantly, the scale derivative
uses `mu`, the multiplier/slope of the bucket representation constraint. It
must not use the full weight derivative, because the explicit perspective
term does not depend directly on `s`.

In the current segment representation, away from kinks and feasibility
boundaries, `mu_i` is `beta_old`: the slope of the active lower-hull segment of
the points `(v_b, xi_b)`. The existing `beta_star` is instead

\[
\texttt{beta_star}=\texttt{beta_old}+2T_1w/y^*.
\]

## Optimizer step

For optimizer step `k`, all quantities must be evaluated at the same state:

1. form `v_k = s_k q`;
2. run the LSQ fake-quantized forward with `(w_k, s_k)`;
3. solve the METaQ inner/dual problem using `(w_k, v_k)`;
4. form
   \[
   g_w=\nabla_wL_{\rm QAT}+\partial_w\phi,
   \qquad
   g_s=\nabla_sL_{\rm QAT}+\partial_s\phi;
   \]
5. update `w` and `s`; the resulting `v_{k+1}` is used by the next step.

Updating `s` before computing the METaQ multiplier in the same step would mix
two different codebooks and would not be a gradient of the stated objective.

## Non-smooth and boundary cases

The value function is non-smooth when the active lower-hull segment changes,
when `y` reaches 1, or when representability forces
`y = |w| / max_b |v_b|`. At such points the multiplier, and hence the scale
subgradient, may be set-valued. A consistent active-set multiplier is required.

The adjacent lower-hull slope is sufficient on an interior segment, but it is
not the KKT multiplier at an extreme representability boundary. If the positive
or negative extreme bucket `(v_e, xi_e)` is active and
`y=|w|/|v_e|`, stationarity gives

\[
\mu_i=\frac{\xi_e+T_3}{v_e}-T_1v_e.
\]

The perspective knapsack solver must use this value instead of the adjacent
segment slope. This also makes the scale envelope derivative agree with finite
differences at the LSQ clipping boundary. Hull kinks remain non-smooth and admit
a set of valid subgradients.

LSQ clips latent weights outside `[s*Qn, s*Qp]`. To keep the METaQ inner
problem feasible, the joint objective is defined on the same clipped weight:

\[
\phi(\operatorname{clip}(w,sQ_n,sQ_p),s).
\]

For an out-of-range coordinate, the scale derivative therefore contains the
additional chain-rule term `(dphi/dw_clipped) * q_edge`, while its METaQ weight
gradient is zero. This is separate from the direct envelope derivative above.

## Relation to the LSQ gradient

LSQ already supplies a surrogate task-loss gradient for `s`, including its
gradient normalization. The joint update adds the METaQ scale derivative to
that task gradient. The two components must be logged separately and scaled by
their objective coefficients; normalizing the METaQ component merely to match
the task-gradient norm would change the stated objective and should be treated
as a separate heuristic experiment.
