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
    from MiningEnv.envs.demo import DemoTrack
    from MiningEnv.envs.path_planner import TruckPlanner
    from MiningEnv.envs.path_planner import ExcavatorPlanner
except BaseException:
    # When running current file as a file
    from demo import DemoTrack
    from path_planner import TruckPlanner
    from path_planner import ExcavatorPlanner


class MiningEnv(gym.Env):
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
        observation_low = np.array([50, 250, 0, 300, 100, -np.pi/2, -np.pi/2])
        observation_high = np.array([650, 700, 2*np.pi, 600, 200, np.pi/2, np.pi/2])
        self.observation_space = [spaces.Box(observation_low, observation_high, dtype=np.float32),
                                  spaces.Box(observation_low, observation_high, dtype=np.float32)]

        truck_low = np.array([250, 300, np.pi*150/180])
        truck_high = np.array([650, 480, np.pi*210/180])

        excvtr_low = np.array([300, 100, -np.pi/2, -np.pi/2])
        excvtr_high = np.array([600, 200, np.pi/2, np.pi/2])
        self.action_space = [spaces.Box(truck_low, truck_high, dtype=np.float32),
                             spaces.Box(excvtr_low, excvtr_high, dtype=np.float32)]

        self.seed()
        self.viewer = None
        self.state = None

        # Demo track
        self.track = DemoTrack()
        self.n = 2
        self.truck_planner = TruckPlanner()
        self.excvtr_planner = ExcavatorPlanner()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Need to add path planner
        truck_trgt, excvtr_trgt = action
        truck_trgt = np.clip(truck_trgt, self.action_space[0].low, self.action_space[0].high)
        excvtr_trgt = np.clip(excvtr_trgt, self.action_space[1].low, self.action_space[1].high)

        print('truck_trgt is {}, excvtr_trgt is {}'.format(truck_trgt, excvtr_trgt))

        truck_current = [self.state[0], self.state[1], self.state[2]]
        excvtr_current = [self.state[3], self.state[4], self.state[5]]
        self.truck_path = self.truck_planner.get_path(truck_current,
                                                      truck_trgt)
        self.excvtr_path = self.excvtr_planner.get_path(excvtr_current,
                                                        excvtr_trgt)

        self.step_counter = 0
        self.cntr += 1

        observation = [self.state, self.state]
        reward = -reeds_shepp.path_length(tuple(truck_current), tuple(
            excvtr_current), self.truck_planner.rho)

        done = False
        if self.cntr >= 20:
            done = True  # reaches time limit
        elif self._can_load_on_truck():
            done = True
            reward = 8000

        info = None

        return observation, [reward, reward], [done, done], info

    def reset(self):
        # Need to figure out the initialization region
        low = np.array([50, 650, -(np.pi*80)/180, 300, 100,
                        (np.pi*80)/180, (np.pi*170)/180])
        high = np.array([100, 700, -(np.pi*100)/180, 350, 120,
                         (np.pi*100)/180, (np.pi*190)/180])
        self.state = self.np_random.uniform(low=low, high=high)
        # self.state = np.concatenate((state, state), axis=None)
        self.cntr = 0  # records step number for termination check
        self.i = 0
        return [self.state, self.state]

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

        self.trucktrans.set_translation(self.state[0], self.state[1])
        self.trucktrans.set_rotation(self.state[2])  # in radian
        self.excvtrplattrans.set_translation(self.state[3], self.state[4])
        self.excvtrplattrans.set_rotation(self.state[5])
        self.excvtrbintrans.set_translation(self.state[3], self.state[4])
        self.excvtrbintrans.set_rotation(self.state[6])

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

    def _translate_angle(self, angle, diff):
        reeds_shepp_version = np.pi * (angle + diff) / 180
        return reeds_shepp_version

    def update_state(self):
        old_state = self.state
        try:
            self.state = np.asarray(self.truck_path[self.step_counter])
        except BaseException:
            self.state = np.asarray(self.truck_path[-1])

        try:
            self.state = np.concatenate((self.state, np.asarray(
                self.excvtr_path[self.step_counter])), axis=None)
        except BaseException:
            self.state = np.concatenate((self.state, np.asarray(
                self.excvtr_path[-1])), axis=None)

        if old_state[6] > np.pi/2:
            self.state = np.concatenate((self.state, old_state[6]-np.pi/200), axis=None)
        else:
            self.state = np.concatenate((self.state, np.pi/2), axis=None)

        self.step_counter += 1

    def _can_load_on_truck(self):
        truck_pos = [self.state[0], self.state[1]]
        truck_ori = self.state[2]
        excvtr_pos = [self.state[3], self.state[4]]

        truck_cntr_d = [math.cos(truck_ori) * 50, math.sin(truck_ori) * 50]
        truck_cntr = [truck_pos[0]-truck_cntr_d[0], truck_pos[1]-truck_cntr_d[1]]
        dis_x = math.pow(truck_cntr[0]-excvtr_pos[0], 2)
        dis_y = math.pow(truck_cntr[1]-excvtr_pos[1], 2)
        dis = math.sqrt(dis_x + dis_y)

        if dis > 155:
            return False
        else:
            # print('--- successfully loaded on truck ---')
            return True


def main():
    # Test rendering effect
    m = MiningEnv()

    for i in range(5):
        done = False
        step = 0
        rewards = 0
        m.reset()
        while not done:
            truck_action = m.action_space[0].sample()
            excvtr_action = m.action_space[1].sample()
            action = [truck_action, excvtr_action]
            observation, reward, done_n, info = m.step(action)
            for _ in range(60):
                step += 1
                m.render()
                m.update_state()
            done = all(done_n)
        print('Episode {}, Reward {}'.format(i, rewards))
    m.close()


if __name__ == "__main__":
    main()
