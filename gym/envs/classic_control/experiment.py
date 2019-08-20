import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering


class ExpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.observation_space = spaces.Box(
            np.array([-200, -200, 0]), np.array([200, 200, 2*math.pi]), dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-math.pi/180]), np.array([math.pi/180]), dtype=np.float32)

        self.seed()
        self.state = None
        self.viewer = None
        self.max_episode_len = 1000

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.state[2] += action
        reward = 1
        info = {}

        return self.state, reward, False, info

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        agent_length = 50
        agent_height = 50

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = self._render_polygon(agent_length, agent_height)
            agent.add_attr(rendering.Transform(translation=(0, -25)))
            agent.set_color(102/255, 102/255, 102/255)
            self.viewer.add_geom(agent)

            self.agenttrans = rendering.Transform()
            agent.add_attr(self.agenttrans)

        self.agenttrans.set_translation(screen_width/2+self.state[0], screen_height/2+self.state[1])
        self.agenttrans.set_rotation(self.state[2])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _render_polygon(self, width, length):
        l, r, t, b = -width/2, width/2, length, 0
        return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])


def main():
    env = ExpEnv()
    for i in range(1000):
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = math.pi/180
            state, reward, done, info = env.step(action)
            step += 1
            env.render()

            if step > env.max_episode_len:
                break

    env.close()


if __name__ == "__main__":
    main()
