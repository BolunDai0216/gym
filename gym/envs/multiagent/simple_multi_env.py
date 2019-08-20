import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import math
import numpy as np


class SimpleMultiEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self):
        self.observation_space = [spaces.Box(np.array([-20, -20, 0]), np.array([20, 20, 20*math.sqrt(2)]), dtype=np.float32),
                                  spaces.Box(np.array([-20, -20, 0]), np.array([20, 20, 20*math.sqrt(2)]), dtype=np.float32)]

        self.action_space = [spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32),
                             spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)]

        self.seed()
        self.state = None
        self.viewer = None
        self.max_episode_len = 1000
        self.n = 2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action1, action2 = action
        self.state[0][0] += action1[0]
        self.state[0][0] = self._clip_state(self.state[0][0])
        self.state[0][1] += action1[1]
        self.state[0][1] = self._clip_state(self.state[0][1])
        self.state[1][0] += action2[0]
        self.state[1][0] = self._clip_state(self.state[1][0])
        self.state[1][1] += action2[1]
        self.state[1][1] = self._clip_state(self.state[1][1])

        dis1 = math.sqrt(math.pow(self.state[0][0] + 3.5, 2) + math.pow(self.state[0][1], 2))
        reward1 = 1 - math.pow(dis1/(20*math.sqrt(2)), 0.4)
        if dis1 < 2.5:
            reward1 = 10 * reward1

        dis2 = math.sqrt(math.pow(self.state[1][0] - 3.5, 2) + math.pow(self.state[1][1], 2))
        reward2 = 1 - math.pow(dis2/(20*math.sqrt(2)), 0.4)
        if dis2 < 2.5:
            reward2 = 10 * reward2

        dis_12 = math.sqrt(
            math.pow(self.state[0][0] - self.state[1][0], 2) + math.pow(self.state[0][1] - self.state[1][1], 2))

        self.state[0][2] = dis_12
        self.state[1][2] = dis_12

        done = False

        if dis_12 < 6:
            reward1 += 10
            reward2 += 10

        if dis_12 < 5.8:
            reward1 += 1000
            reward2 += 1000
            done = True

        info = {'agent1': None, 'agent2': None}

        return self.state, [reward1, reward2], [done, done], info

    def reset(self):
        self.state = [self.observation_space[0].sample(),
                      self.observation_space[1].sample()]
        print('Initial State: {}'.format(self.state))
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        agent_length = 10
        agent_height = 10

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            goal1 = rendering.make_circle(25)
            goal1.set_color(1, 0, 0)
            goal1.add_attr(rendering.Transform(translation=(235, 200)))
            self.viewer.add_geom(goal1)

            goal2 = rendering.make_circle(25)
            goal2.set_color(0, 0, 1)
            goal2.add_attr(rendering.Transform(translation=(165, 200)))
            self.viewer.add_geom(goal2)

            agent1 = self._render_polygon(agent_length, agent_height)
            agent1.set_color(102/255, 102/255, 102/255)
            agent1.add_attr(rendering.Transform(translation=(200, 200)))
            self.viewer.add_geom(agent1)

            self.agenttrans1 = rendering.Transform()
            agent1.add_attr(self.agenttrans1)

            agent2 = self._render_polygon(agent_length, agent_height)
            agent2.set_color(0, 0, 0)
            agent2.add_attr(rendering.Transform(translation=(200, 200)))
            self.viewer.add_geom(agent2)

            self.agenttrans2 = rendering.Transform()
            agent2.add_attr(self.agenttrans2)

        self.agenttrans1.set_translation(10 * self.state[0][0], 10 * self.state[0][1])
        self.agenttrans2.set_translation(10 * self.state[1][0], 10 * self.state[1][1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _render_polygon(self, width, length):
        l, r, t, b = -width/2, width/2, length, 0
        return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

    def _clip_state(self, state):
        if state > 20:
            return 20
        elif state < -20:
            return -20
        else:
            return state


def main():
    env = SimpleMultiEnv()
    env.reset()
    for _ in range(1000):
        action1 = env.action_space[0].sample()
        action2 = env.action_space[1].sample()
        env.step([action1, action2])
        env.render()
    env.close()


if __name__ == "__main__":
    main()
