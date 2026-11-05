"""Render a scripted demonstration or an actual learned policy."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
from .env import ProbePushEnv
from .evaluate import baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint'); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output', default='artifacts/preview.mp4')
    p.add_argument('--human', action='store_true')
    args = p.parse_args()
    policy = baseline
    label = 'Scripted baseline (untrained)'
    if args.checkpoint:
        from .sac import SAC
        policy = SAC().load(args.checkpoint).predict
        label = 'Learned SAC policy'
    env = ProbePushEnv(render_mode='human' if args.human else 'rgb_array')
    frames = []
    def annotate(frame, caption):
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        draw.rectangle((0,0,960,65), fill=(18,25,35))
        draw.text((18,12), f'CERG input-learning bench | {label}', fill='white')
        draw.text((18,36), caption, fill=(110,215,255))
        return im
    try:
        obs, _ = env.reset(seed=args.seed)
        frames.append(annotate(env.render(), 'Two probes, then energy bounds + rear-face contact selection'))
        done = False
        while not done:
            phase = env.phase
            env.frames = []
            obs, _, done, _, info = env.step(policy(obs))
            caption = f'Probe {phase+1}/2' if phase < 2 else 'Push with selected energy envelope'
            for f in env.frames:
                frames.append(annotate(f, caption))
        caption = f"Success: {info['is_success']} | positive work {info['work_j']:.3f} J | peak contact {info['peak_force_n']:.2f} N"
        frames.extend([annotate(env.render(), caption)] * 25)
        info['contact_point_block_m'] = env.contact_point()
    finally:
        env.close()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out.with_suffix('.png'))
    with tempfile.TemporaryDirectory() as tmp:
        for j, frame in enumerate(frames):
            frame.save(Path(tmp)/f'{j:05d}.png')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','25','-i',f'{tmp}/%05d.png',
                        '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out)], check=True)
    frames[0].resize((720,480)).save(out.with_suffix('.gif'), save_all=True,
        append_images=[f.resize((720,480)) for f in frames[1:]], duration=40, loop=0)
    out.with_suffix('.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2)); print(out)


if __name__ == '__main__':
    main()
