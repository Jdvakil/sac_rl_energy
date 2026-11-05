"""Active probing and energy-budget selection on RoboHive."""
from gymnasium.envs.registration import register

register(id="CERGPush-v0", entry_point="cerg_push.env:ProbePushEnv")
