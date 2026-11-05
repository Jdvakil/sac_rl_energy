# Learning reference inputs for contact-aware pushing

A working RoboHive/MuJoCo proof of concept: a blue planar pusher probes an orange block twice, then a SAC reference manager chooses `(Emin, Emax, rear-face contact point)` to push the block into a green target. Uses your existing **conda environment `robohive`**, its PyTorch, and `/home/jaydv/code/robohive` without modifying that repository or installing packages.

**This is a learning and simulation bench, not an implementation of CERG.** No CERG equations/code were supplied. The included controller uses a positive actuator-work budget and a feedforward force. Its energy variables must be mapped to your CERG's actual storage function and reference law before learned proposals can be used as CERG inputs. Policy learning and empirical confidence do not establish the controller's hard-constraint guarantees.

## See the sample

The generated [preview video](artifacts/preview.mp4), [animated preview](artifacts/preview.gif), and [scene image](artifacts/preview.png) use an explicitly labeled **scripted baseline**, not a trained agent. The preview telemetry is in `artifacts/preview.json`.

```bash
conda activate robohive
cd /home/jaydv/code/test_rl
MUJOCO_GL=egl python -m cerg_push.preview --seed 7
```

To open the interactive MuJoCo viewer on a desktop, run `python -m cerg_push.preview --human`. For headless rendering use `MUJOCO_GL=egl`; avoid that setting for the desktop viewer. RGB frames/video rendering is disabled during training.

## Train SAC

```bash
conda activate robohive
cd /home/jaydv/code/test_rl
python -m cerg_push.train --steps 60000 --seed 0 --run-dir runs/sac_seed0
```

`--steps` counts high-level decisions; a full episode contains **three decisions** and many physics steps. Defaults are CPU, one Torch thread, two 128-unit hidden layers, a 100k transition buffer, 1,500 warmup decisions, batch size 256, gamma 0.99, tau 0.005, learning rate 3e-4, and learned entropy temperature. CPU is a reasonable starting point for this small network. `--device cuda` is optional. Simulation is not GPU vectorized.

`latest.pt` includes replay, optimizer, and RNG states. Resume at a new episode boundary:

```bash
python -m cerg_push.train --resume runs/sac_seed0/latest.pt --run-dir runs/sac_seed0 --steps 30000
```

Resume restores learning state but not an interrupted physics episode or CUDA RNG state; it is not bitwise continuation. `progress.jsonl` records validation performance every 3,000 decisions. `best.pt` is selected by the lower endpoint of the success confidence interval on fixed validation seeds starting at 50,000. A run shorter than the evaluation interval only produces `latest.pt`. Training is left to you; the included short smoke run only verifies the code path.

SAC is the initial choice because this is a small continuous action space and replay can reuse relatively expensive probe/push transitions. Its entropy objective supports exploration; it does **not** provide confidence estimates. PPO failure alone does not establish algorithm superiority. This bench changes the task formulation as well as the algorithm, so compare multiple training seeds and the scripted baseline before concluding SAC is better.

## Task, observation, and action contract

- Physics: 500 Hz, two probes of at most 0.4 s each and one push of at most 5 s. The block can translate in xyz and yaw; roll/pitch are constrained for this simple planar experiment. Vertical motion creates genuine table normal forces and sliding friction.
- Unknown parameters are resampled per episode: mass 0.25–0.75 kg, sliding friction 0.15–0.60, and lateral center-of-mass offset ±1 cm. Goal distance is 0.14–0.22 m. The COM offset makes contact choice relevant even for a geometrically symmetric block.
- Policy observations are 23 floats: phase/2, goal, block x/y/yaw, block x/y/yaw velocity, cumulative positive actuator work, and two seven-value probe records. Each probe record contains commanded force/4, lateral contact offset/0.03, dx, dy, yaw, work J, peak block–pusher force/20. Hidden mass/friction/COM are absent from observations and policy-facing info.
- Four normalized actions in `[-1,1]`: Emin, Emax gap, contact offset, probe intensity. In the first two decisions, contact and probe intensity are used; the energy outputs are ignored. In the final decision the two energy outputs and contact are used; probe intensity is ignored. Every phase's full action vector is stored in replay.
- Decode: `Emin = 0.02 + 0.78*u0` J; `Emax = Emin + 0.02 + (1.98-Emin)*u1` J, where `u=(action+1)/2`; contact y is ±0.03 m on the rear block face. Thus `0 < Emin < Emax <= 2 J`. Contact is expressed in the **block frame**, `[-0.035, y, 0]` m, not world coordinates.
- Probe intensity maps to 0.4–4 N. Each probe has a fixed 0.12 J budget, and switches to damping after 1.2 cm forward displacement or 0.3 s. Probes can move/rotate the block; that state carries into the push. Their work and peaks count in episode metrics.
- The pusher is repositioned beside the block between phases while detached. This is an abstract positioning operation, not a learned robot trajectory; repositioning work is excluded. A full-arm implementation must account for this motion.

The surrogate sets a slowly advancing reference and `Fx = 120*(xref-x) - 5*vx + Emin/goal_distance`. It clips both motor forces to ±12 N and stops active commands when predicted positive work reaches Emax. Work integrates positive force-times-displacement per actuator at each physics step. This is a **phase-local cumulative work budget**, not instantaneous kinetic energy, stored spring energy, or a CERG energy tank. Probing budgets are separate; total episode work can exceed the final Emax.

Success requires final x error <1.5 cm, lateral error <3.5 cm, yaw <0.2 rad, and speed <2.5 cm/s. Reward favors success, low final error/yaw, and lower work/Emax; monitored violations incur a penalty and end the episode. Limits monitored every physics step are contact force >20 N, work >budget+0.002 J, |y|>0.15 m, or |yaw|>0.6 rad. Force spikes may occur before termination; this is a monitor, not an invariant safety filter. The actual actuator force saturation is enforced in the MuJoCo model. Finite table extent and simulation numerical effects also preclude claiming all physical hard constraints are guaranteed.

## Evaluate confidence and export a proposal

Evaluate the frozen checkpoint on **independent test seeds**, then compare against the baseline and outside the training parameter range:

```bash
python -m cerg_push.evaluate --checkpoint runs/sac_seed0/best.pt --episodes 200 --output artifacts/sac_test.json
python -m cerg_push.evaluate --episodes 200 --output artifacts/baseline_test.json
python -m cerg_push.evaluate --checkpoint runs/sac_seed0/best.pt --episodes 200 --ood --output artifacts/sac_ood.json
MUJOCO_GL=egl python -m cerg_push.preview --checkpoint runs/sac_seed0/best.pt --output artifacts/learned.mp4
python -m cerg_push.propose --checkpoint runs/sac_seed0/best.pt --evaluation artifacts/sac_test.json --output artifacts/proposal.json
```

Reports include success counts and a two-sided 95% **Clopper–Pearson binomial interval**, violations, work, force, per-episode constants, and local contact coordinates. The policy is deterministic during evaluation. Test seeds default to 100,000; OOD trials use mass 0.75–1.0 kg and friction 0.60–0.85. Compare policies on the same seeds. Do not tune on the final test set; use validation for tuning. Repeat training with seeds 1 and 2 (separate run directories) to assess training variability.

The exported proposal uses a fresh episode and two probes. It includes the measured policy success interval, checkpoint hash, and a reliability gate: default lower success bound >=0.90 and zero observed violations. A failed gate still exports the proposal as **unqualified**, with `empirical_reliability_gate_passed=false`; consumers must check this field. Passing it is neither a per-proposal probability nor a safety certification. OOD reports cannot qualify an in-distribution proposal. No claim of the “highest confidence” is possible before training and independent evaluation, and statistical confidence alone cannot establish distribution-shift robustness.

## Connect the actual CERG

`ProbePushEnv(controller=your_callback)` calls `your_callback(model, data, proposal, target_xy)` every physics step during the final push (the two probes retain their force protocol) and expects a two-element force command. `proposal` contains `emin_j`, `emax_j`, and `contact_y_m`. This callback replaces the PD force command, but the surrogate work budget remains as an outer simulation monitor. Use `env.contact_point()` after the final rollout or decode the final action before rollout for the chosen block-local contact point.

To make results CERG-specific, replace the surrogate budget/reference code inside `_rollout` with the real storage function and CERG update, define exactly what Emin/Emax mean, include all relevant hard constraints, and retrain/evaluate against that controller. The current callback alone does not make those energy semantics equivalent. You also need to decide whether energy means an instantaneous bound, a supplied-work budget, or a task-level resource, and how force bounds follow from your system's geometry/compliance. This project deliberately makes no universal joule-to-newton conversion.

## Validation and files

```bash
MUJOCO_GL=egl python -m unittest discover -s tests -v
python -m cerg_push.train --steps 300 --warmup 256 --eval-every 300 --eval-episodes 3 --run-dir artifacts/smoke_run
```

Tests cover Gymnasium compatibility/seeding, hidden-parameter exclusion, physical friction effects, a successful symmetric baseline push, actuator saturation, energy decoding, binomial boundaries, SAC updates, and checkpoint restoration. The smoke training verifies collection, optimization, evaluation, and saving; it does not show convergence.

`env.py` holds the task, `scene.xml` the geometry, `sac.py` the learner, `train.py` training/checkpoint selection, `evaluate.py` confidence evaluation, `propose.py` proposal export, and `preview.py` rendering. The local RoboHive `SimScene` supplies physics and its renderer supplies images. The table texture is loaded from SimHive's Fetch assets. Set `ROBOHIVE_ASSETS` if the SimHive directory moves.

References: [SAC paper](https://arxiv.org/abs/1801.01290), [SAC implementation notes and continuous-space support](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html), [MuJoCo contact force conventions](https://mujoco.readthedocs.io/en/2.1.4/programming.html). This project implements SAC directly in PyTorch; it does not use Stable-Baselines3.
