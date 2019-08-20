#!/usr/bin/python3
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import math
from pdb import set_trace
import reeds_shepp


try:
    # When running as a package
    from gym.envs.classic_control.demo import DemoTrack
    from gym.envs.classic_control.path_planner import TruckPlanner
    from gym.envs.classic_control.path_planner import ExcavatorPlanner
    from gym.envs.classic_control.collision_check import Rectangle
    from gym.envs.classic_control.collision_check import CollisionCheck
    from gym.envs.classic_control.collision_check import Contains
except BaseException:
    # When running current file as a file
    from demo import DemoTrack
    from path_planner import TruckPlanner
    from path_planner import ExcavatorPlanner
    from collision_check import Rectangle
    from collision_check import CollisionCheck
    from collision_check import Contains


class SingleMiningEnv(gym.Env):
    """
    Description:
        A truck and an excavator is tasked to move a pile of minerals out of
        the load site. The truck and the excavator both starts with a random
        position and orientation in a designated region. The goal is to
        coordinate both the truck and the excavator to reach a position where
        the hand of the excavator is right above the container of the truck.

    Observations:
        Type: Box(14)
        Num        Observation                              Min            Max
        0          Truck x-position for truck               50             650
        1          Truck y-position for truck               250            700
        2          Truck orientation for truck(deg)         0              360
        3          Excav. x-position for truck              300            600
        4          Excav. y-position for truck              100            200
        5          Excav. orientation for truck(deg)        -90            90
        6          Excav. hand orientation for truck(deg)   -90            90
        7          Truck x-position for excav.              50             650
        8          Truck y-position for excav.              250            700
        9          Truck orientation for excav.(deg)        0              360
        10         Excav. x-position for excav.             300            600
        11         Excav. y-position for excav.             100            200
        12         Excav. orientation for excav.(deg)       -90            90
        13         Excav. hand orientation for excav.(deg)  -90            90

    Actions:
        Type: Box(7)
        Num        Action                              Min            Max
        0          Truck loading x-position            250            650
        1          Truck loading y-position            300            480
        2          Truck loading orientation(deg)      135            225
        3          Excav. loading x-position           300            600
        4          Excav. loading y-position           100            200
        5          Excav. loading orientation(deg)     -90            90
        6          Excav. hand orientation(deg)        -90            90

    Reward:
        Reward is -reeds_shepp_curve_length for every step taken, including
        the termination step, and 7000 for successfully loading on the truck.

    Starting State:
        All observations are assigned a uniform random value corresponding to
        the random initial state.

    Episode Termination:
        The hand of the excavator is above the truck bin. Then the system
        executes dumping and signals the truck to exit the loading site.

        or

        Episode length is greater than 20

    Solved Requirements(beta):
        Considered solved when the average reward is greater than or equal to
        0 over 100 consecutive trials.

    Confusions:
        1. Does the excavator and the truck operate in completely different
           regions of the loading site, thus diminishes the possibility of them
           colliding.
        2. How should I construct the path planner.

    Places that may cause confusions:
        1. The truck in gym environment is pointing down when no rotation
           (0 deg) is added, however the excavator is pointing up when (0 deg)
           no rotation is added. The code for Reeds-Shepp path generation uses
           a different set of angle definitions, in their definition 0 deg is
           pointing right.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.observation_low = np.array([50, 250, -1, -1, 300, 100, -1, -1, -1, -1])
        self.observation_high = np.array([650, 700, 1, 1, 600, 200, 1, 1, 1, 1])
        self.observation_range = self.observation_high - self.observation_low
        self.observation_space = spaces.Box(np.zeros(10), np.ones(10), dtype=np.float32)

        # [truck_x, truck_y]
        self.truck_low = np.array([250, 300])
        self.truck_high = np.array([650, 320])
        self.truck_range = self.truck_high - self.truck_low
        # [e_x, e_y]
        self.excvtr_low = np.array([300, 180])
        self.excvtr_high = np.array([350, 200])
        self.excvtr_range = self.excvtr_high - self.excvtr_low
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.if_render = 0
        self.n = 1
        self.episode = 0

        self.truck_planner = TruckPlanner()
        self.excvtr_planner = ExcavatorPlanner()
        self.success_counter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Input: list(np.array([1x2]), np.array([1x2]))
        Output: None
        Use: Perform a step in the simulation environment.
        """
        # scaled target state
        scl_truck_trgt = (action + np.ones(2))/2
        scl_excvtr_trgt = np.array([0.5, 0.5])

        # unscaled target state
        truck_trgt = self.truck_low + np.multiply(scl_truck_trgt, self.truck_range)
        excvtr_trgt = self.excvtr_low + np.multiply(scl_excvtr_trgt, self.excvtr_range)

        # radian state
        rd_state = self._get_radian_state()
        truck_state = rd_state[:3].tolist()
        truck_trgt = truck_trgt.tolist()
        excvtr_state = rd_state[3:-1].tolist()
        excvtr_trgt = excvtr_trgt.tolist()

        # get reeds-shepp path
        self.truck_path = self.truck_planner.get_path(truck_state, truck_trgt)
        self.excvtr_path = self.excvtr_planner.get_path(excvtr_state, excvtr_trgt)
        # set_trace()

        self.step_counter = 0
        # self.counter += 1
        info = {}

        # execute first 60 points on the path if collides terminate the episode
        for _ in range(60):
            if self.if_render:
                self.render()
            obsv, flag = self._update_state()
            if flag:
                self.reward += -1000
                print('episode: {}, reward: {}'.format(self.episode, self.reward))
                return obsv, -1000, True, info

        # get current scaled state
        scaled_state = self._get_scaled_state()
        observation = scaled_state

        # get current radian state
        rd_state = self._get_radian_state()
        truck_state = rd_state[:3].tolist()

        excvtr_pos = rd_state[3:5]
        arm_ori = rd_state[-1]
        arm_pos = excvtr_pos + np.array([130*math.cos(arm_ori), 130*math.sin(arm_ori)])
        arm_state = np.concatenate((arm_pos, np.pi/2), axis=None)

        # calculate reward
        reward = -reeds_shepp.path_length(tuple(truck_state), tuple(
            arm_state), self.truck_planner.rho)

        # check for termination
        done = False
        if self._can_load_on_truck():
            done = True
            reward = 8000
            # self.success_counter += 1
            # print('loads on truck, count: {}'.format(self.success_counter))

        self.reward += reward
        if done:
            print('episode: {}, reward: {}'.format(self.episode, self.reward))
        return observation, reward, done, info

    def reset(self):
        # Define unscaled initial state region
        # Initial state for truck is between [50, 650 -100deg] ~ [100, 700, -80deg]
        # For the excavator it is [300, 100, 80deg, 170deg] ~ [350, 120, 100deg, 190deg]
        low = np.array([50, 650, -1, math.cos(math.pi*(-100)/180),
                        300, 100, math.sin(math.pi*(80)/180),
                        math.cos(math.pi*(100)/180),
                        math.sin(math.pi*(190)/180), -1])
        high = np.array([100, 700, math.sin(math.pi*(-80)/180),
                         math.cos(math.pi * (-80)/180), 350, 120, 1,
                         math.cos(math.pi*(80)/180),
                         math.sin(math.pi*(170)/180),
                         math.cos(math.pi*(190)/180)])

        # Sample initial state
        self.state = self.np_random.uniform(low=low, high=high)
        # Scale sampled state
        scaled_state = self._get_scaled_state()
        obsv = scaled_state

        # records step number for termination check
        # self.counter = 0
        self.episode += 1
        self.reward = 0

        return obsv

    def render(self, mode='human', demo=0, render=1):
        screen_width = 600
        screen_height = 800

        truck_container_width = 60
        truck_container_length = 100
        truck_head_width = 50
        truck_head_length = 20

        excvtr_platform_width = 80
        excvtr_platform_length = 100
        excvtr_track_width = 15
        excvtr_track_length = 80
        excvtr_bin_size = 40  # Square
        excvtr_arm_width = 12
        excvtr_arm_length = 100
        excvtr_hand_width = 40
        excvtr_hand_length = 20

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            minerals = rendering.make_circle(10)
            minerals.set_color(1, 171/255, 64/255)
            minerals.add_attr(rendering.Transform(translation=(150, 100)))
            self.viewer.add_geom(minerals)

            self.trucktrans = rendering.Transform()
            self.excvtrplattrans = rendering.Transform()
            self.excvtrbintrans = rendering.Transform()

            cntnr = self._render_polygon(truck_container_length, truck_container_width)
            cntnr.set_color(102/255, 102/255, 102/255)
            cntnr.add_attr(rendering.Transform(translation=(0, -truck_container_width/2)))
            cntnr.add_attr(self.trucktrans)
            self.viewer.add_geom(cntnr)

            hd = self._render_polygon(truck_head_length, truck_head_width)
            hd.set_color(191/255, 144/255, 0)
            hd.add_attr(rendering.Transform(translation=(truck_head_length /
                                                         2 + truck_container_length/2, -truck_head_width/2)))
            hd.add_attr(self.trucktrans)
            self.viewer.add_geom(hd)

            excvtr_plat = self._render_polygon(excvtr_platform_length, excvtr_platform_width)
            excvtr_plat.set_color(241/255, 194/255, 50/255)
            excvtr_plat.add_attr(rendering.Transform(
                translation=(excvtr_bin_size/2, -excvtr_platform_width/2)))
            excvtr_plat.add_attr(self.excvtrplattrans)
            self.viewer.add_geom(excvtr_plat)

            excvtr_track_l = self._render_polygon(excvtr_track_length, excvtr_track_width)
            excvtr_track_l.set_color(67/255, 67/255, 67/255)
            excvtr_track_l.add_attr(rendering.Transform(translation=(
                excvtr_bin_size/2,
                excvtr_platform_width/2)))
            excvtr_track_l.add_attr(self.excvtrplattrans)
            self.viewer.add_geom(excvtr_track_l)

            excvtr_track_r = self._render_polygon(excvtr_track_length, excvtr_track_width)
            excvtr_track_r.set_color(67/255, 67/255, 67/255)
            excvtr_track_r.add_attr(rendering.Transform(translation=(
                excvtr_bin_size/2,
                -excvtr_platform_width/2-excvtr_track_width)))
            excvtr_track_r.add_attr(self.excvtrplattrans)
            self.viewer.add_geom(excvtr_track_r)

            excvtr_bin = self._render_polygon(excvtr_bin_size, excvtr_bin_size)
            excvtr_bin.set_color(164/255, 194/255, 244/255)
            excvtr_bin.add_attr(rendering.Transform(translation=(0, -excvtr_bin_size/2)))
            excvtr_bin.add_attr(self.excvtrbintrans)
            self.viewer.add_geom(excvtr_bin)

            excvtr_arm = self._render_polygon(excvtr_arm_length, excvtr_arm_width)
            excvtr_arm.set_color(204/255, 204/255, 204/255)
            excvtr_arm.add_attr(rendering.Transform(
                translation=(excvtr_arm_length/2+excvtr_bin_size/2, -excvtr_arm_width/2)))
            excvtr_arm.add_attr(self.excvtrbintrans)
            self.viewer.add_geom(excvtr_arm)

            excvtr_hand = self._render_polygon(excvtr_hand_length, excvtr_hand_width)
            excvtr_hand.set_color(90/255, 90/255, 90/255)
            excvtr_hand.add_attr(rendering.Transform(
                translation=(excvtr_arm_length+excvtr_bin_size/2+excvtr_hand_length/2, -excvtr_hand_width/2)))
            excvtr_hand.add_attr(self.excvtrbintrans)
            self.viewer.add_geom(excvtr_hand)

        if render:
            # Render planned path
            truck_xys, truck_trgt = self._get_trajectory_list(self.truck_path)
            excvtr_xys, excvtr_trgt = self._get_trajectory_list(self.excvtr_path)

            self.truck_track = rendering.make_polyline(truck_xys)
            self.truck_track.set_color(0, 0, 1)
            self.truck_track.set_linewidth(4)
            self.viewer.add_geom(self.truck_track)

            truck_destination = rendering.make_circle(10)
            truck_destination.set_color(0, 0, 1)
            truck_destination.add_attr(rendering.Transform(translation=(
                truck_trgt[0], truck_trgt[1])))
            self.viewer.add_geom(truck_destination)

            self.excvtr_track = rendering.make_polyline(excvtr_xys)
            self.excvtr_track.set_color(1, 0, 0)
            self.excvtr_track.set_linewidth(4)
            self.viewer.add_geom(self.excvtr_track)

            excvtr_destination = rendering.make_circle(10)
            excvtr_destination.set_color(1, 0, 0)
            excvtr_destination.add_attr(rendering.Transform(translation=(
                excvtr_trgt[0], excvtr_trgt[1])))
            self.viewer.add_geom(excvtr_destination)

        rd_state = self._get_radian_state()

        self.trucktrans.set_translation(rd_state[0], rd_state[1])
        self.trucktrans.set_rotation(rd_state[2])  # in radian
        self.excvtrplattrans.set_translation(rd_state[3], rd_state[4])
        self.excvtrplattrans.set_rotation(rd_state[5])
        self.excvtrbintrans.set_translation(rd_state[3], rd_state[4])
        self.excvtrbintrans.set_rotation(rd_state[6])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _render_polygon(self, width, length):
        l, r, t, b = -width/2, width/2, length, 0
        return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

    def _get_trajectory_list(self, trajectory):
        x = []
        y = []

        for point in trajectory:
            x.append(point[0])
            y.append(point[1])

        xys = list(zip(x, y))
        destination = [x[-1], y[-1]]
        return xys, destination

    def _update_state(self):
        old_state = self._get_radian_state()
        try:
            rd_state = np.asarray(self.truck_path[self.step_counter])
        except BaseException:
            rd_state = np.asarray(self.truck_path[-1])

        try:
            rd_state = np.concatenate((rd_state, np.asarray(
                self.excvtr_path[self.step_counter])), axis=None)
        except BaseException:
            rd_state = np.concatenate((rd_state, np.asarray(
                self.excvtr_path[-1])), axis=None)

        if old_state[6] > np.pi/2:
            rd_state = np.concatenate((rd_state, old_state[6]-np.pi/200), axis=None)
        else:
            rd_state = np.concatenate((rd_state, np.pi/2), axis=None)

        self.state = self._get_sin_cos_state(rd_state)
        scaled_state = self._get_scaled_state()
        obsv = scaled_state
        flag = self._check_collision()

        self.step_counter += 1

        return obsv, flag

    def _can_load_on_truck(self):
        """
        Input: None
        Output: bool(1x1)
        Use: Gives a true-false response for whether the current state of the
             truck and the excavator is suitable for loading.
        """

        rd_state = self._get_radian_state()
        truck_pos = rd_state[:2]
        truck_ori = rd_state[2]
        rect = Rectangle(truck_pos, 60, 40, truck_ori)
        excvtr_pos = rd_state[3:5]
        arm_ori = rd_state[-1]
        arm_pos = excvtr_pos + np.array([130*math.cos(arm_ori), 130*math.sin(arm_ori)])
        contains = Contains(arm_pos[0], arm_pos[1], rect)
        return contains.contains()

    def _get_radian_state(self):
        """
        Input: None
        Output: np.array([1x7])
        Use: Returns the state of the environment where the orientation of the
             truck and the excavator are given in radians.
        """

        truck_pos = [self.state[0], self.state[1]]
        truck_ang_rad = math.atan2(self.state[2], self.state[3])
        excvtr_pos = [self.state[4], self.state[5]]
        excvtr_plat_ang_rad = math.atan2(self.state[6], self.state[7])
        excvtr_arm_ang_rad = math.atan2(self.state[8], self.state[9])

        radian_state = truck_pos + [truck_ang_rad] + \
            excvtr_pos + [excvtr_plat_ang_rad, excvtr_arm_ang_rad]
        return np.array(radian_state)

    def _get_scaled_state(self):
        """
        Input: None
        Output: np.array([1x10])
        Use: Returns the state of the environment after scaling them all
             between 0 and 1.
        """
        scaled_state = (self.state - self.observation_low)/self.observation_range
        return scaled_state

    def _get_sin_cos_state(self, rd_state):
        """
        Input: np.array([1x7])
        Output: np.array([1x10])
        Use: Returns the state of the environment after changing the radian
             states into a sin-cos representation.
        """
        truck_pos = rd_state[:2]
        truck_sin_cos = np.array([np.sin(rd_state[2]), np.cos(rd_state[2])])
        excvtr_pos = rd_state[3:5]
        excvtr_plat_sin_cos = np.array([np.sin(rd_state[5]), np.cos(rd_state[5])])
        excvtr_arm_sin_cos = np.array([np.sin(rd_state[6]), np.cos(rd_state[6])])
        sin_cos_state = np.concatenate(
            (truck_pos, truck_sin_cos, excvtr_pos, excvtr_plat_sin_cos, excvtr_arm_sin_cos), axis=None)
        return sin_cos_state

    def _check_collision(self):
        rd_state = self._get_radian_state()
        rect1 = Rectangle(rd_state[:2], 120, 60, rd_state[2], 10)
        rect2 = Rectangle(rd_state[3:5], 100, 80, rd_state[5], 20)
        ccheck = CollisionCheck(rect1, rect2)
        return ccheck.if_collide()


def main():
    # Test rendering effect
    m = SingleMiningEnv()

    for i in range(60000):
        done = False
        rewards = 0
        m.reset()
        while not done:
            truck_action = m.action_space.sample()
            action = truck_action
            observation, reward, done, info = m.step(action)
            rewards += reward
    m.close()


if __name__ == "__main__":
    main()
