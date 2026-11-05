"""Gymnasium task backed by RoboHive SimScene (not a CERG implementation).

A learned reference manager probes twice, then chooses a work envelope and
contact offset. The controller below is an explicit, replaceable surrogate.
"""
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from robohive.physics.sim_scene import SimScene
import mujoco


@dataclass(frozen=True)
class Proposal:
    emin_j: float
    emax_j: float
    contact_y_m: float


def decode(action):
    a = (np.clip(np.asarray(action, dtype=float), -1, 1) + 1) / 2
    emin = .02 + .78 * a[0]
    return Proposal(float(emin), float(emin + .02 + (2.0 - emin - .02) * a[1]),
                    float((a[2] * 2 - 1) * .03))


class ProbePushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 25}

    def __init__(self, render_mode=None, ood=False, controller=None):
        super().__init__()
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError(render_mode)
        self.render_mode, self.ood = render_mode, ood
        self.controller = controller
        xml = Path(__file__).with_name('scene.xml').read_text()
        root = Path(os.environ.get('ROBOHIVE_ASSETS', '/home/jaydv/code/robohive/robohive/simhive'))
        texture = root / 'fetch_sim/texture/marble.png'
        if not texture.exists():
            raise FileNotFoundError(f'{texture}: set ROBOHIVE_ASSETS to your SimHive directory')
        # Reuse SimHive's table texture; analytic box supplies the collision surface.
        xml = xml.replace('<!-- ROBOHIVE_TABLE_MESH -->', f'<texture name="rh_marble" type="2d" file="{texture}"/><material name="rh_surface" texture="rh_marble" rgba=".5 .6 .7 1"/>')
        xml = xml.replace('material="surface"', 'material="rh_surface"')
        self.sim = SimScene.get_sim(xml)
        self.model, self.data = self.sim.model.ptr, self.sim.data.ptr
        self.block = self.model.body('block').id
        self.block_geom = self.model.geom('block_geom').id
        self.pusher_geom = self.model.geom('pusher_geom').id
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (23,), dtype=np.float32)
        self.frames = None
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)
        mass_range = (.75, 1.0) if self.ood else (.25, .75)
        mu_range = (.6, .85) if self.ood else (.15, .6)
        self.mass = float(options.get('mass', self.np_random.uniform(*mass_range)))
        self.mu = float(options.get('friction', self.np_random.uniform(*mu_range)))
        if self.mass <= 0 or self.mu < 0:
            raise ValueError('mass must be positive and friction nonnegative')
        self.com_y = float(options.get('com_y', self.np_random.uniform(-.01, .01)))
        self.model.body_ipos[self.block, 1] = self.com_y
        self.model.body_mass[self.block] = self.mass
        self.model.body_inertia[self.block] = self.mass / 3 * np.array([.045**2+.025**2, .035**2+.025**2, .035**2+.045**2])
        self.model.geom_friction[self.block_geom, 0] = self.mu
        self.model.geom_friction[self.model.geom('table').id, 0] = self.mu
        mujoco.mj_setConst(self.model, self.data)
        self.goal = float(options.get('goal', self.np_random.uniform(.14, .22)))
        self.model.site_pos[self.model.site('goal').id, 0] = self.goal
        self.history = np.zeros((2, 7))
        self.phase, self._done = 0, False
        self.total_work = self.peak_force = 0.
        self.proposal = None
        self.force_trace = []
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def _obs(self):
        # Measurable state + finite probe history, never latent mass/friction.
        return np.r_[self.phase / 2, self.goal, self.data.qpos[[0,1,3]], self.data.qvel[[0,1,3]],
                     self.total_work, self.history.ravel()].astype(np.float32)

    def _position_pusher(self, offset):
        yaw = self.data.qpos[3]
        # Point on the rear block face transformed from block to world frame.
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        target = self.data.qpos[:2] + rot @ np.array([-.050, offset])
        self.data.qpos[4:6] = target - np.array([-.058, 0])
        self.data.qvel[4:6] = 0
        mujoco.mj_forward(self.model, self.data)
        return target

    def _contact_force(self):
        force = np.zeros(6)
        total = 0.
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if {c.geom1, c.geom2} == {self.block_geom, self.pusher_geom}:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += np.linalg.norm(force[:3])
        return float(total)

    def _rollout(self, proposal, probe_force=None):
        target = self._position_pusher(proposal.contact_y_m)
        start = self.data.qpos[:2].copy()
        work, peak, success, violation = 0., 0., False, False
        dt = self.model.opt.timestep
        count = 200 if probe_force is not None else 2500
        for t in range(count):
            x = self.data.body('pusher').xpos[0]
            vx = self.data.qvel[4]
            if probe_force is not None:
                fx = probe_force - 3 * vx
                if self.data.qpos[0] - start[0] > .012 or t >= 150:
                    fx = -8 * vx
            else:
                # Surrogate: Emin sets nominal feedforward work per distance;
                # Emax limits positive actuator work, NOT instantaneous tank energy.
                target[0] = min(target[0] + .09 * dt, self.goal - .047)
                fx = 120 * (target[0] - x) - 5 * vx + proposal.emin_j / self.goal
            fy = 150 * (target[1] - self.data.body('pusher').xpos[1]) - 5 * self.data.qvel[5]
            cmd = np.clip([fx, fy], -12, 12)
            if self.controller is not None and probe_force is None:
                cmd = np.clip(self.controller(self.model, self.data, proposal, target.copy()), -12, 12)
            # Predict next-step velocity conservatively to stop active work before budget.
            estimate = np.sum(np.abs(cmd) * (np.abs(self.data.qvel[4:6]) + np.abs(cmd) / .15 * dt)) * dt
            if work + estimate >= proposal.emax_j:
                cmd[:] = 0
            old_velocity = self.data.qvel[4:6].copy()
            self.data.ctrl[:] = cmd
            mujoco.mj_step(self.model, self.data)
            increment = float(np.maximum(cmd * (old_velocity + self.data.qvel[4:6]) / 2, 0).sum() * dt)
            work += increment
            force = self._contact_force()
            peak = max(peak, force)
            self.force_trace.append(force)
            # A detected violation stops a rollout; this is not a force guarantee.
            violation = force > 20 or work > proposal.emax_j + .002 or abs(self.data.qpos[1]) > .15 or abs(self.data.qpos[3]) > .6
            success = (abs(self.data.qpos[0] - self.goal) < .015 and abs(self.data.qpos[1]) < .035
                       and abs(self.data.qpos[3]) < .2 and np.linalg.norm(self.data.qvel[:2]) < .025)
            if self.frames is not None and t % 20 == 0:
                self.frames.append(self.render())
            if self.render_mode == 'human' and t % 20 == 0:
                self.sim.renderer.render_to_window()
            if violation or (success and probe_force is None):
                break
        self.data.ctrl[:] = 0
        self.total_work += work
        self.peak_force = max(self.peak_force, peak)
        return self.data.qpos[:2] - start, work, peak, success, violation

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset before stepping a finished episode')
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Expected four finite actions')
        proposal = decode(action)
        probe = self.phase < 2
        if probe:
            force = .4 + (np.clip(action[3], -1, 1) + 1) / 2 * 3.6
            # Probes have an independent, fixed budget, not learned Emin/Emax.
            proposal = Proposal(.02, .12, proposal.contact_y_m)
            displacement, work, peak, success, violation = self._rollout(proposal, force)
            self.history[self.phase] = [force / 4, proposal.contact_y_m / .03, *displacement,
                                       self.data.qpos[3], work, peak / 20]
            reward = -work - .05
        else:
            self.proposal = proposal
            displacement, work, peak, success, violation = self._rollout(proposal)
            error = abs(self.goal - self.data.qpos[0])
            reward = (10 * float(success) - 20 * error - 8 * abs(self.data.qpos[1])
                      - abs(self.data.qpos[3]) - self.total_work - .25 * proposal.emax_j)
        if violation:
            reward -= 15
        self.phase += 1
        self._done = bool(violation or self.phase == 3)
        info = dict(is_success=bool(success and not probe and not violation), violation=bool(violation),
                    work_j=float(self.total_work), peak_force_n=float(self.peak_force),
                    goal_error_m=float(abs(self.goal-self.data.qpos[0])), phase=self.phase)
        if self.proposal:
            info['proposal'] = asdict(self.proposal)
        return self._obs(), float(reward), self._done, False, info

    def contact_point(self):
        """Proposal contact in block-local coordinates (rear face)."""
        return None if self.proposal is None else [-.035, self.proposal.contact_y_m, 0.]

    def render(self):
        return self.sim.renderer.render_offscreen(width=960, height=640, camera_id='overview')

    def close(self):
        renderer = self.sim.renderer
        if renderer._renderer is not None:
            renderer._renderer.close()
            renderer._renderer = None
        if renderer._window is not None:
            renderer._window.close()
            renderer._window = None
