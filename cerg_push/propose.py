"""Export selected constants with independently measured policy reliability."""
import argparse
import hashlib
import json
from pathlib import Path
from .env import ProbePushEnv, decode
from .sac import SAC
from .evaluate import confidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--evaluation', required=True, help='Independent evaluate.py report for this checkpoint')
    p.add_argument('--min-success-lower-bound', type=float, default=.9)
    p.add_argument('--seed', type=int, default=200000)
    p.add_argument('--output', default='artifacts/proposal.json')
    args = p.parse_args()
    if not 0 <= args.min_success_lower_bound <= 1:
        p.error('min-success-lower-bound must be in [0,1]')
    report = json.loads(Path(args.evaluation).read_text())
    digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if report.get('checkpoint_sha256') != digest:
        p.error('Evaluation does not match checkpoint')
    if report['ood'] or report['seed_start'] < 100000:
        p.error('Use a held-out in-distribution test with seeds >=100000, not validation seeds')
    lower = confidence(report['successes'], report['episodes'])[0]
    policy = SAC().load(args.checkpoint)
    env = ProbePushEnv()
    try:
        obs,_ = env.reset(seed=args.seed)
        for _ in range(2):
            obs,_,done,_,info = env.step(policy.predict(obs))
            if done:
                raise RuntimeError('Probe violated a monitored limit; no proposal exported')
        proposal = decode(policy.predict(obs))
        result = dict(Emin_J=proposal.emin_j, Emax_J=proposal.emax_j,
                      contact_point_block_m=[-.035, proposal.contact_y_m, 0.],
                      policy_success_ci95=report['success_ci95'],
                      empirical_reliability_gate_passed=bool(lower >= args.min_success_lower_bound and report['violations'] == 0),
                      controller='work_budget_surrogate', requires_cerg_energy_semantics_validation=True,
                      checkpoint_sha256=digest, seed=args.seed,
                      note='Confidence is for average success under the test distribution, not this contact or hardware safety.')
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
