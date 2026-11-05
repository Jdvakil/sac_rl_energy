"""Independent episode-level success confidence; no policy certainty claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from .env import ProbePushEnv


def baseline(obs):
    # Deliberately simple central-contact comparator, not an optimized policy.
    return np.array([.2, .5, 0, -.2 if obs[0] < .25 else .4], np.float32)


def confidence(k, n, level=.95):
    if n < 1 or not 0 <= k <= n:
        raise ValueError('Need 0 <= successes <= episodes and episodes > 0')
    tail = (1-level)/2
    return [float(beta.ppf(tail, k, n-k+1)) if k else 0.,
            float(beta.ppf(1-tail, k+1, n-k)) if k < n else 1.]


def evaluate(policy, episodes=100, seed=100000, ood=False, records=False):
    env = ProbePushEnv(ood=ood)
    results = []
    try:
        for j in range(episodes):
            obs, _ = env.reset(seed=seed+j)
            done = False
            while not done:
                obs, _, done, _, info = env.step(policy(obs))
            info['seed'] = seed+j
            info['contact_point_block_m'] = env.contact_point()
            results.append(info)
    finally:
        env.close()
    n = len(results); k = sum(x['is_success'] for x in results)
    report = dict(episodes=n, successes=k, success_rate=k/n, success_ci95=confidence(k,n),
                  violations=sum(x['violation'] for x in results),
                  mean_work_j=float(np.mean([x['work_j'] for x in results])),
                  max_force_n=max(x['peak_force_n'] for x in results),
                  ood=ood, seed_start=seed)
    if records:
        report['episodes_detail'] = results
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint'); p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=100000); p.add_argument('--ood', action='store_true')
    p.add_argument('--output', default='artifacts/evaluation.json')
    args = p.parse_args()
    policy = baseline
    if args.checkpoint:
        from .sac import SAC
        policy = SAC().load(args.checkpoint).predict
    report = evaluate(policy, args.episodes, args.seed, args.ood, True)
    report['policy'] = 'baseline' if not args.checkpoint else 'sac'
    if args.checkpoint:
        report['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'episodes_detail'}, indent=2))


if __name__ == '__main__':
    main()
