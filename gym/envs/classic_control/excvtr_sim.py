import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

MAX_SPEED = 20
MAX_ACCELERATION = 2
TIME_INTERVAL = 0.1


class ExcvtrSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.observation_space = spaces.Box(
            np.array([-200, -200, -20, -20]),
            np.array([200, 200, 20, 20]),
            dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1]), np.array([+1, +1]), dtype=np.float32)

        self.seed()
        self.state = None
        self.viewer = None
        self.max_episode_len = 400

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pre_dis = math.sqrt(math.pow(self.state[0], 2) + math.pow(self.state[1], 2))/10

        x_accel = action[0] * MAX_ACCELERATION
        y_accel = action[1] * MAX_ACCELERATION

        x_vel = self.state[2] + x_accel * TIME_INTERVAL
        y_vel = self.state[3] + y_accel * TIME_INTERVAL

        x_vel = np.clip(x_vel, -20, 20)
        y_vel = np.clip(y_vel, -20, 20)

        x = self.state[0] + (self.state[2] + 0.5 * x_accel * TIME_INTERVAL) * TIME_INTERVAL
        y = self.state[1] + (self.state[3] + 0.5 * y_accel * TIME_INTERVAL) * TIME_INTERVAL

        x = np.clip(x, -200, 200)
        y = np.clip(y, -200, 200)

        self.state = np.array([x, y, x_vel, y_vel])

        dis = math.sqrt(math.pow(self.state[0], 2) + math.pow(self.state[1], 2))/10
        x_vel_disc = math.pow(
            1-np.amax([math.fabs(x_vel)/20, 0.1]), 1/np.amax([dis/20*math.sqrt(2), 0.1]))
        y_vel_disc = math.pow(
            1-np.amax([math.fabs(y_vel)/20, 0.1]), 1/np.amax([dis/20*math.sqrt(2), 0.1]))
        reward = (pre_dis - dis) * x_vel_disc * y_vel_disc

        done = False
        if dis < 1.5:
            reward = 2 * x_vel_disc * y_vel_disc
            done = True

        info = {}
        obsv = np.array([self.state[0]/200, self.state[1]/200,
                         self.state[2]/20, self.state[3]/20])
        return obsv, reward, done, info

    def reset(self):
        # Random initialization with 0 velocity
        self.state = self.observation_space.sample()
        self.state[2] = 0
        self.state[3] = 0

        obsv = np.array([self.state[0]/200, self.state[1]/200,
                         self.state[2]/20, self.state[3]/20])

        return obsv

    def render(self, mode='human', view='state_space'):
        screen_width = 400
        screen_height = 400

        agent_length = 10
        agent_height = 10

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            if view == 'state_space':
                goal = rendering.make_circle(10)
                goal.set_color(1, 171/255, 64/255)
                goal.add_attr(rendering.Transform(translation=(screen_width/2, screen_height/2)))
                self.viewer.add_geom(goal)

                agent = self._render_polygon(agent_length, agent_height)
                agent.set_color(102/255, 102/255, 102/255)
                agent.add_attr(rendering.Transform(translation=(0, -agent_height/2)))
                self.viewer.add_geom(agent)

                self.agenttrans = rendering.Transform()
                agent.add_attr(self.agenttrans)

            elif view == 'excavator':
                pass
        self.agenttrans.set_translation(
            screen_width/2 + self.state[0], screen_height/2 + self.state[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _render_polygon(self, width, length):
        l, r, t, b = -width/2, width/2, length, 0
        return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])


def main():
    env = ExcvtrSimEnv()
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        env.step(action)
        env.render()
    env.close()


if __name__ == "__main__":
    main()
