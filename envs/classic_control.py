import gym
import cv2
from gym import wrappers
import numpy as np

class ClassicControlEnv(gym.Env):

    def __init__(self, env_name, seed=0, max_steps=1000, n_frames=1, action_repeat=1, image_size=(84,84)):
        self._env = gym.make(env_name)
        self._env.seed(seed)
        self.max_steps = max_steps
        self.action_repeat = action_repeat
        self.image_size = image_size
        self.observation = np.zeros((n_frames, image_size[0], image_size[1]))
        if self._env.spec:
            self.spec = self._env.spec
        else:
            self.spec = None

    def reset(self):
        self.t = 0
        state = self._env.reset()
        image = self._env.render(mode='rgb_array')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
        self.observation[-1] = image
        return self.observation

    def step(self, action):
        reward = 0
        for i in range(self.action_repeat):
            state, reward_i, done, info = self._env.step(action)
            image = self._env.render(mode='rgb_array')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
            reward += reward_i
            self.t += 1
            terminal = done or self.t == self.max_steps
            if done:
                break
        self.observation = np.roll(self.observation, -1, axis=0)
        self.observation[-1] = image
        return self.observation, reward, terminal, {'concepts': state}

    def close(self):
        return self._env.close()

    def render(self, mode):
        return self._env.render(mode)

    @property
    def observation_space(self):
        image_box = gym.spaces.Box(0, 255, (self.image_size[0], self.image_size[1]), dtype=np.uint8)
        return image_box

    @property
    def action_space(self):
        return self._env.action_space