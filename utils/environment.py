import numpy as np
import gym
# from .atari_wrapper import make_wrap_atari


def wrap_env(env_name):
    env = gym.make(env_name)
    if len(env.observation_space.shape) > 1:
        if 'Pong' in env_name:
            return PongWrapper(env)
        # else:
            # return make_wrap_atari(env_name)
    else:
        return env


class PongWrapper(gym.Wrapper):
    """Wrap Pong env so
        - Observation is preprocessed.
        - An episode ends when one get one point.

    Args:
        env: Pong env.
    """
    def __init__(self, env):
        super(PongWrapper, self).__init__(env)
        self._prev_obs = 0
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(80, 80, 1))
        self._real_done = True

    def reset(self):
        if self._real_done:
            self.env.reset()
            self._real_done = False

        return np.zeros((80, 80, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        obs = obs.astype(float)

        # crop
        obs = obs[35:195]

        # down sample
        obs = obs[::2, ::2, :1]

        # get last color channel
        obs = obs[:, :, 0:1]

        # remove background
        obs[obs == 144] = 0
        obs[obs == 109] = 0

        # change color to white
        obs[obs != 0] = 1

        # get difference from previous state
        processed = obs - self._prev_obs

        # store current state
        self._prev_obs = obs

        # store if really done
        self._real_done = done
        done = reward != 0

        return processed, reward, done, info
