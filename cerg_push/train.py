import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .env import ProbePushEnv
from .sac import SAC
from .evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description='Train the reference manager, not a low-level robot policy')
    p.add_argument('--steps', type=int, default=60000, help='Additional high-level decisions (3 per full episode)')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cpu')
    p.add_argument('--run-dir', default='runs/sac_seed0'); p.add_argument('--resume')
    p.add_argument('--warmup', type=int, default=1500)
    p.add_argument('--eval-every', type=int, default=3000)
    p.add_argument('--eval-episodes', type=int, default=30)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.eval_episodes < 1:
        p.error('steps and evaluation counts must be positive')
    torch.set_num_threads(args.threads)
    out = Path(args.run_dir); out.mkdir(parents=True, exist_ok=True)
    env = ProbePushEnv(); env.action_space.seed(args.seed)
    agent = SAC(seed=args.seed, device=args.device)
    if args.resume:
        agent.load(args.resume, resume=True)
    (out/'config.json').write_text(json.dumps(vars(args), indent=2))
    obs, _ = env.reset(seed=args.seed + agent.steps)
    best = float('-inf'); start = time.monotonic()
    if (out/'best_metrics.json').exists():
        best = json.loads((out/'best_metrics.json').read_text())['success_ci95'][0]
    try:
        with (out/'progress.jsonl').open('a') as log:
            for t in range(args.steps):
                act = env.action_space.sample() if agent.steps < args.warmup else agent.sample(obs)
                no, rew, done, _, info = env.step(act)
                agent.add(obs, act, rew, no, done)
                loss = agent.update() if agent.size >= 256 and agent.steps >= args.warmup else {}
                obs = no
                if done:
                    obs, _ = env.reset()
                if (t+1) % 100 == 0:
                    print(json.dumps(dict(step=agent.steps, seconds=round(time.monotonic()-start,1), **loss)), flush=True)
                if agent.steps % args.eval_every == 0:
                    # Fixed validation set for model selection; final test uses separate seeds.
                    metrics = evaluate(agent.predict, args.eval_episodes, seed=50000)
                    metrics['step'] = agent.steps
                    log.write(json.dumps(metrics)+'\n'); log.flush()
                    print('validation', json.dumps(metrics), flush=True)
                    if metrics['success_ci95'][0] > best:
                        best = metrics['success_ci95'][0]
                        agent.save(out/'best.pt', replay=False)
                        (out/'best_metrics.json').write_text(json.dumps(metrics, indent=2))
                    agent.save(out/'latest.pt')
    finally:
        agent.save(out/'latest.pt'); env.close()
    print(f'Saved {out}/latest.pt. Evaluate on independent test seeds before exporting proposals.')


if __name__ == '__main__':
    main()
