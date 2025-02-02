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
        self.observation_space = [spaces.Box(np.array([-20, -20]), np.array([20, 20]), dtype=np.float32),
                                  spaces.Box(np.array([-20, -20]), np.array([20, 20]), dtype=np.float32)]

        self.action_space = [spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32),
                             spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)]

        self.seed()
        self.state = None
        self.viewer = None
        self.max_episode_len = 250
        self.n = 2
        self.goal = [np.array([-3.5, 0]), np.array([3.5, 0])]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action1, action2 = action

        pre_dis1 = math.sqrt(
            math.pow(self.state[0][0]-self.goal[0][0], 2) + math.pow(self.state[0][1]-self.goal[0][1], 2))
        pre_dis2 = math.sqrt(
            math.pow(self.state[1][0]-self.goal[1][0], 2) + math.pow(self.state[1][1]-self.goal[1][1], 2))

        action_ratio = 1
        action1 = np.clip(action1, -1, 1)
        action2 = np.clip(action2, -1, 1)
        self.state[0][0] += action_ratio * action1[0]
        self.state[0][0] = self._clip_state(self.state[0][0])
        self.state[0][1] += action_ratio * action1[1]
        self.state[0][1] = self._clip_state(self.state[0][1])
        self.state[1][0] += action_ratio * action2[0]
        self.state[1][0] = self._clip_state(self.state[1][0])
        self.state[1][1] += action_ratio * action2[1]
        self.state[1][1] = self._clip_state(self.state[1][1])

        dis1 = math.sqrt(math.pow(self.state[0][0]-self.goal[0]
                                  [0], 2) + math.pow(self.state[0][1]-self.goal[0][1], 2))
        dis2 = math.sqrt(math.pow(self.state[1][0]-self.goal[1]
                                  [0], 2) + math.pow(self.state[1][1]-self.goal[1][1], 2))

        # Improvement reward
        reward1 = pre_dis1 - dis1
        done1 = False
        if dis1 < 1:
            reward1 = 2
            done1 = True

        # Improvement reward
        reward2 = pre_dis2 - dis2
        done2 = False
        if dis2 < 1:
            reward2 = 2
            done2 = True

        # Check for collision
        agent_dis = math.sqrt(math.pow(self.state[0][0]-self.state[1]
                                       [0], 2) + math.pow(self.state[0][1]-self.state[1][1], 2))
        if agent_dis < 1:
            reward1 = -10
            reward2 = -10
            done1 = True
            done2 = True

        info = {'agent1': None, 'agent2': None}

        obsv = [self.state[0]/20, self.state[1]/20]

        return obsv, [reward1, reward2], [done1, done2], info

    def reset(self):
        self.state = [self.observation_space[0].sample(),
                      self.observation_space[1].sample()]
        obsv = [self.state[0]/20, self.state[1]/20]
        return obsv

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        agent_length = 10
        agent_height = 10

        ratio = 10

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            goal1 = rendering.make_circle(25)
            goal1.set_color(1, 0, 0)
            goal1.add_attr(rendering.Transform(translation=(screen_width/2+ratio *
                                                            self.goal[0][0], screen_height/2+ratio*self.goal[0][1])))
            self.viewer.add_geom(goal1)

            goal2 = rendering.make_circle(25)
            goal2.set_color(0, 0, 1)
            goal2.add_attr(rendering.Transform(translation=(screen_width/2+ratio *
                                                            self.goal[1][0], screen_height/2+ratio*self.goal[1][1])))
            self.viewer.add_geom(goal2)

            agent1 = self._render_polygon(agent_length, agent_height)
            agent1.set_color(102/255, 102/255, 102/255)
            agent1.add_attr(rendering.Transform(translation=(screen_width/2, screen_height/2)))
            self.viewer.add_geom(agent1)

            self.agenttrans1 = rendering.Transform()
            agent1.add_attr(self.agenttrans1)

            agent2 = self._render_polygon(agent_length, agent_height)
            agent2.set_color(0, 0, 0)
            agent2.add_attr(rendering.Transform(translation=(screen_width/2, screen_height/2)))
            self.viewer.add_geom(agent2)

            self.agenttrans2 = rendering.Transform()
            agent2.add_attr(self.agenttrans2)

        self.agenttrans1.set_translation(ratio * self.state[0][0], ratio * self.state[0][1])
        self.agenttrans2.set_translation(ratio * self.state[1][0], ratio * self.state[1][1])
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
