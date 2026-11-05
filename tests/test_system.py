import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from cerg_push.env import ProbePushEnv, decode
from cerg_push.evaluate import baseline, confidence
from cerg_push.sac import SAC


class BenchTests(unittest.TestCase):
    def test_gym_api(self):
        env = ProbePushEnv()
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_hidden_parameters_and_repeatability(self):
        env = ProbePushEnv()
        try:
            low,_ = env.reset(seed=11, options={'mass':.5, 'friction':.2})
            a = baseline(low)
            x1 = env.step(a)[0]
            low2,_ = env.reset(seed=11, options={'mass':.5, 'friction':.2})
            x2 = env.step(a)[0]
            np.testing.assert_array_equal(x1,x2)
            high,_ = env.reset(seed=11, options={'mass':.5, 'friction':.6})
            np.testing.assert_array_equal(low,high)  # no privileged observation
            x3 = env.step(a)[0]
            self.assertGreater(abs(x1[2]-x3[2]), .001)  # friction actually affects physics
            np.testing.assert_array_equal(low,low2)
        finally:
            env.close()

    def test_baseline_and_saturation(self):
        env = ProbePushEnv()
        try:
            obs,_ = env.reset(seed=1, options={'mass':.5,'friction':.3,'goal':.18,'com_y':0})
            for _ in range(3):
                obs,_,done,_,info = env.step(baseline(obs))
                self.assertTrue(np.all(np.abs(env.data.actuator_force) <= 12+1e-6))
                self.assertTrue(np.isfinite(obs).all())
                if done:
                    break
            self.assertTrue(info['is_success'], info)
            self.assertFalse(info['violation'])
            self.assertGreater(info['work_j'],0)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(4))
        finally:
            env.close()

    def test_work_cutoff_changes_outcome(self):
        env = ProbePushEnv()
        try:
            outcomes = []
            for push in [np.array([-1,-1,0,0]), np.array([.2,.5,0,0])]:
                obs,_ = env.reset(seed=1, options={'mass':.5,'friction':.3,'goal':.18,'com_y':0})
                for _ in range(2):
                    obs,_,done,_,info = env.step(baseline(obs))
                    self.assertFalse(done)
                probe_work = env.total_work
                _,_,_,_,info = env.step(push)
                self.assertLessEqual(info['work_j'] - probe_work, decode(push).emax_j + .002)
                outcomes.append(info['is_success'])
            self.assertEqual(outcomes, [False, True])
        finally:
            env.close()

    def test_bounds_confidence(self):
        for _ in range(100):
            p = decode(np.random.uniform(-2,2,4))
            self.assertTrue(0 < p.emin_j < p.emax_j <= 2)
            self.assertLessEqual(abs(p.contact_y_m), .03)
        self.assertEqual(confidence(0,20)[0],0)
        self.assertEqual(confidence(20,20)[1],1)
        self.assertGreater(confidence(200,200)[0], .98)

    def test_sac_update_and_checkpoint(self):
        torch.set_num_threads(1)
        agent = SAC(capacity=300, seed=3)
        for _ in range(270):
            o = np.random.randn(23).astype('float32')
            agent.add(o, np.random.uniform(-1,1,4), 1, o*.9, np.random.rand()<.3)
        before = [x.detach().clone() for x in agent.actor.parameters()]
        losses = agent.update()
        self.assertTrue(all(np.isfinite(list(losses.values()))))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.actor.parameters())))
        obs = np.zeros(23, np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sac.pt'; agent.save(p)
            loaded=SAC(capacity=300).load(p,resume=True)
            np.testing.assert_array_equal(agent.predict(obs), loaded.predict(obs))
            self.assertEqual(loaded.size,270)
            loaded.update()


if __name__ == '__main__':
    unittest.main()
