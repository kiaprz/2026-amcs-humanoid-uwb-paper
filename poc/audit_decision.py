"""Audit full stored decision values; precision is not arithmetic verification."""
import argparse
import hashlib
import json
from pathlib import Path
from check_result import check


def audit(path):
    replay = check(path)
    r = json.loads(path.read_text())
    threshold = r['reference']['f_minus']+r['epsilon_g']
    improvement = r['reference']['f_plus']+r['epsilon_g']
    values = dict(reference_lower=r['reference']['f_minus'], margin=r['epsilon_g'],
                  exclusion_threshold=threshold, UB=r['UB'], exclusion_slack=threshold-r['UB'],
                  improvement_threshold=improvement, LB=r['LB'])
    return dict(values=values,
        decimal_roundtrip={k:format(v,'.17g') if v is not None else None for k,v in values.items()},
        binary64_hex={k:v.hex() if v is not None else None for k,v in values.items()},
        exclusion_holds=r['UB'] <= threshold,
        improvement_holds=r['LB'] is not None and r['LB'] > improvement,
        theorem_outcome=r['theorem_outcome'], replay=replay,
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        formal_certificate=False,
        note='Comparisons use full stored float64 values. Digits and replay do not justify supplied error budgets.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);args=p.parse_args()
    result=audit(args.result)
    (args.result.parent/'decision_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
