# Results and Interpretation

The full deterministic experiment evaluates 1,200 common synthetic market paths
per policy. All policies completed every parent order with zero reported capacity
violations.

| Policy | Mean IS (bps) | 95% CVaR (bps) | Interpretation |
|---|---:|---:|---|
| TWAP | 0.699 | 28.163 | Lowest mean in this synthetic draw, but highest tail cost. |
| Static AC | 0.871 | 23.803 | Lower tail cost than TWAP. |
| Regime-aware MPC | 0.870 | 23.805 | Essentially tied with static AC. |
| Oracle MPC | 0.864 | 23.833 | Full-state diagnostic only; not deployable. |

Paired bootstrap contrasts reinforce the conservative conclusion: the 95% CI for
Regime-aware MPC minus Static AC is `[-0.0029, 0.0011]` bps, so this experiment
does **not** support a claim of a material incremental MPC advantage. Both
optimized policies have substantially lower simulated CVaR than TWAP.

Under persistent severe stress (`P00=0.88`, `P11=0.90`, stressed impact = 36
bps), CVaR rises to 33.537 bps for TWAP and about 29.38–29.40 bps for the two
optimized causal policies. This is a stress diagnostic on synthetic data, not a
forecast or a real-market performance claim.
