import math
import numpy as np

from gym.envs.classic_control.truck_dynamics import Truck
from gym.envs.classic_control.optimal_dumping_path import OptimalDumpingPath

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

MAX_ANGULAR_VEL = math.tan(math.pi/6)
XLIM = [-200, 200]
YLIM = [-200, 0]


class TruckSimVelHalfOptInitEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.observation_space = spaces.Box(
            np.array([XLIM[0], YLIM[0], 0, -20, -MAX_ANGULAR_VEL]),
            np.array([XLIM[1], YLIM[1], 2*math.pi, 20, MAX_ANGULAR_VEL]),
            dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1]), np.array([+1, +1]), dtype=np.float32)

        self.seed()
        self.car = None
        self.state = None
        self.viewer = None
        self.max_episode_len = 450
        self.path = OptimalDumpingPath()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pre_dis = math.sqrt(math.pow(self.state[0]/10, 2) + math.pow(self.state[1]/10, 2))
        pos, angle, vel, ang_vel, FLAG = self.car.step(action[0], action[1])
        angle = angle % (2*math.pi)
        self.state = np.concatenate((np.array(pos), angle), axis=None)
        self.state = np.concatenate((self.state, vel), axis=None)
        self.state = np.concatenate((self.state, ang_vel), axis=None)

        dis = math.sqrt(math.pow(self.state[0]/10, 2) + math.pow(self.state[1]/10, 2))
        reward = pre_dis - dis
        ang_discount = math.pow(
            1-np.amax([self._ang_dis(angle)/math.pi, 0.1]), 1/np.amax([dis/20*math.sqrt(2), 0.1]))
        vel_discount = math.pow(
            1-np.amax([math.fabs(vel)/20, 0.1]), 1/np.amax([dis/20*math.sqrt(2), 0.1]))

        if ang_discount < 0 or vel_discount < 0:
            print('Error')

        reward = ang_discount * vel_discount * reward
        done = False
        if dis < 4:
            reward = 2 * ang_discount
            reward = vel_discount * reward
            done = True
        elif FLAG:
            reward = -1
            done = True

        info = {}
        obsv = np.array([self.state[0]/200, self.state[1]/200,
                         self.state[2]/(2*math.pi), self.state[3]/20,
                         self.state[4]/MAX_ANGULAR_VEL])
        return obsv, reward, done, info

    def reset(self):
        # Optimal Path Initialization
        self.state = self.path.get_init_state()

        self.car = Truck(self.state[0], self.state[1], self.state[2], xlim=XLIM, ylim=YLIM)
        obsv = np.array([self.state[0]/200, self.state[1]/200,
                         self.state[2]/(2*math.pi), self.state[3]/20,
                         self.state[4]/MAX_ANGULAR_VEL])

        return obsv

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 200

        agent_length = 24
        agent_height = 15

        head_length = 5
        head_height = 12

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            goal = rendering.make_circle(40)
            goal.set_color(1, 171/255, 64/255)
            goal.add_attr(rendering.Transform(translation=(screen_width/2, screen_height)))
            self.viewer.add_geom(goal)

            agent = self._render_polygon(agent_length, agent_height)
            agent.set_color(102/255, 102/255, 102/255)
            agent.add_attr(rendering.Transform(translation=(10, -agent_height/2)))
            self.viewer.add_geom(agent)

            head = self._render_polygon(head_length, head_height)
            head.set_color(0, 0, 0)
            head.add_attr(rendering.Transform(translation=(
                10+agent_length/2+head_length/2, -head_height/2)))
            self.viewer.add_geom(head)

            self.agenttrans = rendering.Transform()
            agent.add_attr(self.agenttrans)
            head.add_attr(self.agenttrans)

        self.agenttrans.set_translation(
            screen_width/2 + self.state[0], screen_height + self.state[1])
        self.agenttrans.set_rotation(self.state[2])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _render_polygon(self, width, length):
        l, r, t, b = -width/2, width/2, length, 0
        return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

    def _ang_dis(self, angle):
        if 0 <= angle and angle <= math.pi/2:
            dis = angle + math.pi/2
        elif math.pi/2 < angle and angle <= 3*math.pi/2:
            dis = 3*math.pi/2 - angle
        elif 3*math.pi/2 < angle and angle <= 2*math.pi:
            dis = angle - 3*math.pi/2
        else:
            dis = 0  # numerical issues

        return dis
