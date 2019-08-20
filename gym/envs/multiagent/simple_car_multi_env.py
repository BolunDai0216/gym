import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
from gym.envs.classic_control.truck_dynamics import Truck
import math
import numpy as np

from pdb import set_trace

MAX_ANGULAR_VEL = math.tan(math.pi/6)


class SimpleCarMultiEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # Agent1 and Agent2 both are nonholonomic
        self.observation_space = [spaces.Box(np.array([-200, -200, 0, -20, -MAX_ANGULAR_VEL]), np.array([200, 200, 2*math.pi, 20, MAX_ANGULAR_VEL]), dtype=np.float32),
                                  spaces.Box(np.array([-200, -200, 0, -20, -MAX_ANGULAR_VEL]), np.array([200, 200, 2*math.pi, 20, MAX_ANGULAR_VEL]), dtype=np.float32)]

        self.action_space = [spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32),
                             spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)]

        self.seed()
        self.car1 = None
        self.car2 = None

        self.state = None
        self.viewer = None
        self.max_episode_len = 300
        self.n = 2

        # self.goal = [np.array([-35, 0]), np.array([35, 0])]
        self.goal = [np.array([0, 0]), np.array([0, 0])]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action1, action2 = action

        pre_dis1 = math.sqrt(
            math.pow(self.state[0][0]-self.goal[0][0], 2) + math.pow(self.state[0][1]-self.goal[0][1], 2))/10
        pre_dis2 = math.sqrt(
            math.pow(self.state[1][0]-self.goal[1][0], 2) + math.pow(self.state[1][1]-self.goal[1][1], 2))/10

        action1 = np.clip(action1, -1, 1)
        pos1, angle1, vel1, ang_vel1 = self.car1.step(action1[0], action1[1])
        angle1 = angle1 % (2*math.pi)
        self.state[0] = np.concatenate((np.array(pos1), angle1), axis=None)
        self.state[0] = np.concatenate((self.state[0], vel1), axis=None)
        self.state[0] = np.concatenate((self.state[0], ang_vel1), axis=None)

        action2 = np.clip(action2, -1, 1)
        pos2, angle2, vel2, ang_vel2 = self.car2.step(action2[0], action2[1])
        angle2 = angle2 % (2*math.pi)
        self.state[1] = np.concatenate((np.array(pos2), angle2), axis=None)
        self.state[1] = np.concatenate((self.state[1], vel2), axis=None)
        self.state[1] = np.concatenate((self.state[1], ang_vel2), axis=None)

        dis1 = math.sqrt(math.pow(self.state[0][0]-self.goal[0]
                                  [0], 2) + math.pow(self.state[0][1]-self.goal[0][1], 2))/10
        dis2 = math.sqrt(math.pow(self.state[1][0]-self.goal[1]
                                  [0], 2) + math.pow(self.state[1][1]-self.goal[1][1], 2))/10

        vel_discount1 = math.pow(
            1-np.amax([math.fabs(vel1)/20, 0.1]), 1/np.amax([dis1/20*math.sqrt(2), 0.1]))
        vel_discount2 = math.pow(
            1-np.amax([math.fabs(vel2)/20, 0.1]), 1/np.amax([dis2/20*math.sqrt(2), 0.1]))

        reward1 = (pre_dis1 - dis1) * vel_discount1
        reward2 = (pre_dis2 - dis2) * vel_discount2
        done1 = False
        done2 = False
        if dis1 < 2.5:
            reward1 = 2 * vel_discount1
            done1 = True
        if dis2 < 2.5:
            reward2 = 2 * vel_discount2
            done2 = True
        if dis1 < 2.5 and dis2 < 2.5:
            reward1 = 100 * vel_discount1
            reward2 = 100 * vel_discount2
            done1 = True
            done2 = True

        info = {'agent1': None, 'agent2': None}
        obsv1 = np.array([self.state[0][0]/200, self.state[0][1]/200,
                          self.state[0][2]/(2*math.pi), self.state[0][3]/20,
                          self.state[0][4]/MAX_ANGULAR_VEL])
        obsv2 = np.array([self.state[1][0]/200, self.state[1][1]/200,
                          self.state[1][2]/(2*math.pi), self.state[1][3]/20,
                          self.state[1][4]/MAX_ANGULAR_VEL])
        obsv = [obsv1, obsv2]

        return obsv, [reward1, reward2], [done1, done2], info

    def reset(self):
        self.state = [self.observation_space[0].sample(),
                      self.observation_space[1].sample()]
        self.state[0][0] = 150
        self.state[0][1] = 150
        self.state[1][0] = -150
        self.state[1][1] = -150

        self.car1 = Truck(self.state[0][0], self.state[0][1], self.state[0][2])
        self.car2 = Truck(self.state[1][0], self.state[1][1], self.state[1][2])
        obsv1 = np.array([self.state[0][0]/200, self.state[0][1]/200,
                          self.state[0][2]/(2*math.pi), self.state[0][3]/20,
                          self.state[0][4]/MAX_ANGULAR_VEL])
        obsv2 = np.array([self.state[1][0]/200, self.state[1][1]/200,
                          self.state[1][2]/(2*math.pi), self.state[1][3]/20,
                          self.state[1][4]/MAX_ANGULAR_VEL])
        obsv = [obsv1, obsv2]
        return obsv

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        agent1_length = 24
        agent1_height = 15

        agent2_length = 24
        agent2_height = 15

        ratio = 1

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

            agent1 = self._render_polygon(agent1_length, agent1_height)
            agent1.set_color(102/255, 102/255, 102/255)
            agent1.add_attr(rendering.Transform(translation=(10, -agent1_height/2)))
            self.viewer.add_geom(agent1)

            self.agenttrans1 = rendering.Transform()
            agent1.add_attr(self.agenttrans1)

            agent2 = self._render_polygon(agent2_length, agent2_height)
            agent2.set_color(0, 0, 0)
            agent2.add_attr(rendering.Transform(translation=(10, -agent2_height/2)))
            self.viewer.add_geom(agent2)

            self.agenttrans2 = rendering.Transform()
            agent2.add_attr(self.agenttrans2)

        self.agenttrans1.set_translation(
            screen_width/2 + self.state[0][0], screen_height/2 + self.state[0][1])
        self.agenttrans1.set_rotation(self.state[0][2])
        self.agenttrans2.set_translation(
            screen_width/2 + self.state[1][0], screen_height/2 + self.state[1][1])
        self.agenttrans2.set_rotation(self.state[1][2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _render_polygon(self, width, length):
        l, r, t, b = -width/2, width/2, length, 0
        return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

    def _clip_state(self, state):
        if state > 200:
            return 200
        elif state < -200:
            return -200
        else:
            return state


def main():
    env = SimpleCarMultiEnv()
    env.reset()
    for _ in range(1000):
        action1 = env.action_space[0].sample()
        action2 = env.action_space[1].sample()
        print(action1, action2)
        env.step([action1, action2])
        env.render()
    env.close()


if __name__ == "__main__":
    main()
