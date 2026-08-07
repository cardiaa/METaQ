# Coefficient naming migration

The semantic coefficient names are the canonical code interface from the first
experiment launched after `test_169`:

- `perspective_coeff`: coefficient of the perspective ridge term;
- `sparsity_coeff`: coefficient of the nonzero-mass penalty;
- `entropy_coeff`: coefficient of the relaxed symbol-count term.

Experiments through `test_169` used the historical numbered interface. To
reproduce one of those configurations, translate its logged values as follows:

| Historical log field | Canonical code field |
| --- | --- |
| `T1` | `perspective_coeff` |
| `T2` | `entropy_coeff` |
| `T3` | `sparsity_coeff` |

The paper uses the clearer order: perspective coefficient $T_1$, sparsity
coefficient $T_2$, and entropy coefficient $T_3$. Thus, for a historical run,
the paper values are obtained as $T_1=\texttt{T1}$,
$T_2=\texttt{T3}$, and $T_3=\texttt{T2}$.
