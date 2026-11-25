"""Exact counterexample and nominal a priori tests; no sampled optima."""
from fractions import Fraction as F
from pathlib import Path
import json


def comparisons():
    # Fixed tetrahedral directions sum_i h_i h_i^T = 4I/3.
    # Common constant alpha in [.9,1] cancels in a paired comparison.
    low,high=F(6,5),F(4,3)
    gamma=high-low
    nominal=dict(improvement_nominal_slack_margin_0_59=low-F(8,15)-F(59,100),
                 exclusion_nominal_slack_restricted_margin_0_3=F(3,10)+F(3,10)-F(8,15),
                 optimality_tolerance_0_01_impossible=F(1,100)<gamma)
    return dict(product=dict(index_interval=[str(low),str(high)],absolute_optimum_width=str(gamma),
                             gain_interval=[str(-gamma),str(gamma)],exclusion_margin_0=False),
                paired=dict(channel='same unknown constant alpha in [9/10,1] at every posture',
                            gain_interval=['0','0'],exclusion_margin_0=True),
                nominal_apriori={k:str(v) if isinstance(v,F) else v for k,v in nominal.items()},
                scope='analytic tetrahedral identity and proven sphere extrema; no grid maxima; nominal slacks require accounting for E-/E+ and reference errors before a finite-budget decision')


if __name__=='__main__':
    p=Path('poc/results/reviewer/comparison_models.json');p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(comparisons(),indent=2)+'\n')
