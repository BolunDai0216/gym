import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
from gym.envs.classic_control.truck_dynamics import Truck
from gym.envs.classic_control.excvtr_dynamics import Excavator
import math
import numpy as np

MAX_ANGULAR_VEL = math.tan(math.pi/6)
MAX_DIS = math.sqrt(math.pow(138, 2) + math.pow(210, 2))
MAX_POLAR_DIS = math.sqrt(math.pow(math.pi, 2) + math.pow(math.pi, 2))

MAX_ANG_VEL_E = 0.5
MAX_REL_ANG_VEL = 0.67391304347826086


class TruckExcvtrMultiEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        """
        Observation: [[x_dis, y_dis, angle, lin_vel, ang_vel, vx_rel, vy_rel], [phi_dis, rho_dis, ang_vel, lin_vel, v_rho_rel, v_phi_rel]]
        Action: [[steer, gas], [phi_accel, rho_accel]]
        """
        self.observation_space = [spaces.Box(np.array([-400, -300, 0, -20, -MAX_ANGULAR_VEL, -95, -95]), np.array([400, 100, 2*math.pi, 20, MAX_ANGULAR_VEL, 95, 95]), dtype=np.float32),
                                  spaces.Box(np.array([-math.pi, 0, -MAX_ANG_VEL_E, -5, -25, -MAX_REL_ANG_VEL]), np.array([math.pi, MAX_DIS, MAX_ANG_VEL_E, 5, 25, MAX_REL_ANG_VEL]), dtype=np.float32)]
        self.action_space = [spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32),
                             spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)]

        self.seed()
        self.excvtr_pos = [0, -200]
        self.truck = None
        self.excvtr = None
        self.state = None
        self.viewer = None
        self.max_episode_len = 300
        self.n = 2
        self.truck_pre_pos = None
        self.bckt_pre_pos = None
        self.truck_pos = None
        self.bckt_pos = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        truck_action, excvtr_action = action
        _, flag = self.truck.step(truck_action[0], truck_action[1])
        self.excvtr.step(excvtr_action[0], excvtr_action[1])
        _, _, self.truck_pos, self.bckt_pos = self._get_dis()
        obsv = self._get_obsv()
        rew, done = self._get_rew()
        info = {'Truck': None, 'Excavator': None}

        if flag:
            rew[0] = -1
            done = [True, True]

        return obsv, rew, done, info

    def reset(self):
        self.truck = Truck(138, 10, 0, xlim=[-400, 400], ylim=[-85, 400])
        self.excvtr = Excavator(self.excvtr_pos[0], self.excvtr_pos[1], 0, 60)
        obsv = self._get_obsv()
        _, _, self.truck_pre_pos, self.bckt_pre_pos = self._get_dis()
        return obsv

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 800

        scale = 1

        truck_container_width = 60/scale
        truck_container_length = 100/scale
        truck_head_width = 50/scale
        truck_head_length = 20/scale

        excvtr_platform_width = 80/scale
        excvtr_platform_length = 100/scale
        excvtr_track_width = 15/scale
        excvtr_track_length = 80/scale
        excvtr_bin_size = 40/scale  # Square
        excvtr_hand_width = 40/scale
        excvtr_hand_length = 20/scale

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.trucktrans = rendering.Transform()
            self.excvtrbintrans = rendering.Transform()
            self.excvtrextend = rendering.Transform()
            self.excvtr_extend = rendering.Transform()

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
                translation=(screen_width/2+self.excvtr_pos[0]+excvtr_bin_size/2, screen_height/2+self.excvtr_pos[1]-excvtr_platform_width/2)))
            self.viewer.add_geom(excvtr_plat)

            excvtr_track_l = self._render_polygon(excvtr_track_length, excvtr_track_width)
            excvtr_track_l.set_color(67/255, 67/255, 67/255)
            excvtr_track_l.add_attr(rendering.Transform(translation=(
                screen_width/2+self.excvtr_pos[0]+excvtr_bin_size/2,
                screen_height/2+self.excvtr_pos[1]+excvtr_platform_width/2)))
            self.viewer.add_geom(excvtr_track_l)

            excvtr_track_r = self._render_polygon(excvtr_track_length, excvtr_track_width)
            excvtr_track_r.set_color(67/255, 67/255, 67/255)
            excvtr_track_r.add_attr(rendering.Transform(translation=(
                screen_width/2+self.excvtr_pos[0]+excvtr_bin_size/2,
                screen_height/2+self.excvtr_pos[1]-excvtr_platform_width/2-excvtr_track_width)))
            self.viewer.add_geom(excvtr_track_r)

            excvtr_bin = self._render_polygon(excvtr_bin_size, excvtr_bin_size)
            excvtr_bin.set_color(164/255, 194/255, 244/255)
            excvtr_bin.add_attr(rendering.Transform(translation=(0, -excvtr_bin_size/2)))
            excvtr_bin.add_attr(self.excvtrbintrans)
            self.viewer.add_geom(excvtr_bin)

            excvtr_hand = self._render_polygon(excvtr_hand_length, excvtr_hand_width)
            excvtr_hand.set_color(90/255, 90/255, 90/255)
            excvtr_hand.add_attr(rendering.Transform(
                translation=(excvtr_bin_size/2+excvtr_hand_length/2, -excvtr_hand_width/2)))
            excvtr_hand.add_attr(self.excvtr_extend)
            excvtr_hand.add_attr(self.excvtrextend)
            self.viewer.add_geom(excvtr_hand)

        truck_pos, truck_angle, truck_vel, truck_ang_vel = self.truck.get_state()
        excvtr_angle, excvtr_len, excvtr_ang_vel, excvtr_len_vel = self.excvtr.get_state()

        self.trucktrans.set_translation(screen_width/2+truck_pos[0], screen_height/2+truck_pos[1])
        self.trucktrans.set_rotation(truck_angle)  # in radian

        self.excvtr_extend.set_translation(excvtr_len, 0)
        self.excvtrextend.set_translation(
            screen_width/2+self.excvtr_pos[0], screen_height/2+self.excvtr_pos[1])
        self.excvtrextend.set_rotation(excvtr_angle)

        self.excvtrbintrans.set_translation(
            screen_width/2+self.excvtr_pos[0], screen_height/2+self.excvtr_pos[1])
        self.excvtrbintrans.set_rotation(excvtr_angle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _render_polygon(self, width, length):
        l, r, t, b = -width/2, width/2, length, 0
        return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

    def _get_obsv(self):
        truck_pos, truck_angle, truck_vel, truck_ang_vel = self.truck.get_state()
        excvtr_angle, excvtr_len, excvtr_ang_vel, excvtr_len_vel = self.excvtr.get_state()

        # Distance between truck and excavator platform
        cart_dis = [truck_pos[0]-self.excvtr_pos[0], truck_pos[1]-self.excvtr_pos[1]]
        # Distance between truck and excavator platform in polar coordinates [phi, rho]
        bin_polar_dis = self.cart2pol(cart_dis[0], cart_dis[1])
        # Distance between truck and bucket in polar coordinates [phi, rho]
        polar_dis = [excvtr_angle-bin_polar_dis[0], excvtr_len-bin_polar_dis[1]]
        # Distance between truck and bucket in cartesian coordinates
        cart_truck_bin_dis = self.pol2cart(polar_dis[1], polar_dis[0])

        bckt_cart_vel_x, bckt_cart_vel_y = self.polvel2cartvel(
            excvtr_len_vel, excvtr_ang_vel, excvtr_len, excvtr_angle)

        vx = truck_vel * math.cos(truck_angle)
        vy = truck_vel * math.sin(truck_angle)
        truck_pol_vel_rho, truck_pol_vel_phi = self.cartvel2polvel(vx, vy, cart_dis[0], cart_dis[1])

        truck_obsv = np.array([cart_truck_bin_dis[0]/400, cart_truck_bin_dis[1]/600,
                               truck_angle/2*math.pi, truck_vel/20,
                               truck_ang_vel/MAX_ANGULAR_VEL,
                               (vx-bckt_cart_vel_x)/95, (vy-bckt_cart_vel_y)/95])
        excvtr_obsv = np.array([polar_dis[0]/math.pi, polar_dis[1]/MAX_DIS,
                                excvtr_ang_vel/MAX_ANG_VEL_E, excvtr_len_vel/5,
                                (excvtr_len_vel-truck_pol_vel_rho)/25,
                                (excvtr_ang_vel-truck_pol_vel_phi)/MAX_REL_ANG_VEL])
        return [truck_obsv, excvtr_obsv]

    def _get_rew(self):
        done1 = False
        done2 = False
        # truck_pre_dis, excvtr_pre_dis, truck_bin_pre_pos, bin_truck_pre_pos = pre_dis
        truck_dis, excvtr_dis, truck_bin_pos, bin_truck_pos = self._get_dis()

        # This ensures the max reward is about 20 for each agent each episode
        # base_truck_rew = (truck_pre_dis - truck_dis)/18
        # base_excvtr_rew = 20*(excvtr_pre_dis - excvtr_dis)/(math.sqrt(2)*math.pi)

        # if truck_dis < 120:
        #     base_truck_rew = 4

        # Pure team reward
        # truck_bin_pre_dis = math.sqrt(
        #     math.pow(truck_bin_pre_pos[0], 2)+math.pow(truck_bin_pre_pos[1], 2))
        # truck_bin_dis = math.sqrt(math.pow(truck_bin_pos[0], 2)+math.pow(truck_bin_pos[1], 2))
        # base_truck_rew = (truck_bin_pre_dis - truck_bin_dis)/18
        # base_excvtr_rew = 20*(excvtr_pre_dis - excvtr_dis)/(math.sqrt(2)*math.pi)

        polar_trgt = math.sqrt(math.pow(10*math.pi/180, 2) + math.pow(math.pi*10/MAX_DIS, 2))
        cart_trgt_coord = self.pol2cart(10, math.pi/18)
        cart_trgt = math.sqrt(math.pow(cart_trgt_coord[0], 2) + math.pow(cart_trgt_coord[1], 2))
        cart_dis = math.sqrt(math.pow(truck_bin_pos[0], 2) + math.pow(truck_bin_pos[1], 2))

        # Individual Reward
        base_excvtr_rew = self.coord2dis(self.bckt_pre_pos - self.truck_pre_pos) - \
            self.coord2dis(self.bckt_pos - self.truck_pre_pos)
        base_truck_rew = self.coord2dis(self.truck_pre_pos - self.bckt_pre_pos) - \
            self.coord2dis(self.truck_pos - self.bckt_pre_pos)

        if cart_dis < cart_trgt:
            base_truck_rew = 4
            done1 = True

        if excvtr_dis < polar_trgt:
            base_excvtr_rew = 4
            done2 = True

        truck_pos, truck_angle, truck_vel, truck_ang_vel = self.truck.get_state()
        excvtr_angle, excvtr_len, excvtr_ang_vel, excvtr_len_vel = self.excvtr.get_state()

        # truck_ang_discount = math.pow(
        #     1-np.amax([self._ang_dis(truck_angle)/math.pi, 0.1]), 1/np.amax([cart_dis/MAX_DIS, 0.1]))

        truck_ang_discount = 1
        truck_vel_discount = math.pow(
            1-np.amax([math.fabs(truck_vel)/20, 0.1]), 1/np.amax([cart_dis/MAX_DIS, 0.1]))
        truck_rew = base_truck_rew * truck_ang_discount * truck_vel_discount

        excvtr_ang_discount = math.pow(
            1-np.amax([math.fabs(excvtr_ang_vel)/MAX_ANG_VEL_E, 0.1]), 1/np.amax([excvtr_dis/MAX_POLAR_DIS, 0.1]))
        excvtr_len_discount = math.pow(
            1-np.amax([math.fabs(excvtr_len_vel)/5, 0.1]), 1/np.amax([excvtr_dis/MAX_POLAR_DIS, 0.1]))
        excvtr_rew = base_excvtr_rew * math.sqrt(excvtr_ang_discount * excvtr_len_discount)

        self.bckt_pre_pos = self.bckt_pos
        self.truck_pre_pos = self.truck_pos

        done = [done1, done2]

        return [truck_rew, excvtr_rew], done

    def _get_dis(self):
        truck_pos, truck_angle, truck_vel, truck_ang_vel = self.truck.get_state()
        excvtr_angle, excvtr_len, excvtr_ang_vel, excvtr_len_vel = self.excvtr.get_state()

        # Distance between truck and excavator platform [d]
        cart_dis = math.sqrt(
            math.pow(truck_pos[0]-self.excvtr_pos[0], 2) + math.pow(truck_pos[1]-self.excvtr_pos[1], 2))
        # Distance between truck and excavator platform in polar coordinates [phi, rho]
        bin_polar_dis = self.cart2pol(
            truck_pos[0]-self.excvtr_pos[0], truck_pos[1]-self.excvtr_pos[1])
        # ~/models/truck_excvtr_0814/
        # Distance between truck and excavator bucket in polar coordinates [d]
        polar_dis = math.sqrt(math.pow(
            excvtr_angle-bin_polar_dis[0], 2) + math.pow(math.pi*(excvtr_len-bin_polar_dis[1])/MAX_DIS, 2))

        # ~/models/truck_excvtr_0814_v2/
        # polar_dis = math.sqrt(math.pow(excvtr_angle-3*math.pi/4, 2) +
        #                       math.pow(math.pi*(excvtr_len-120)/MAX_DIS, 2))

        # [x, y] of bin
        bin_pos = np.array(self.excvtr_pos) + np.array(self.pol2cart(excvtr_len, excvtr_angle))
        # [x, y] of truck-bin difference
        truck_bin_pos = np.array(truck_pos) - bin_pos
        # [phi, rho] of bin-truck difference
        bin_truck_pos = self.cart2pol(-truck_bin_pos[0], -truck_bin_pos[1])

        return [cart_dis, polar_dis, truck_bin_pos, bin_truck_pos]

    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return [phi, rho]

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return [x, y]

    def _ang_dis(self, angle):
        angle = angle % (2*math.pi)
        if 0 <= angle and angle <= math.pi/2:
            dis = math.pi/2 - angle
        elif math.pi/2 < angle and angle <= 3*math.pi/2:
            dis = angle - math.pi/2
        elif 3*math.pi/2 < angle and angle < 2*math.pi:
            dis = 5*math.pi/2 - angle
        else:
            dis = 0
        return dis

    def polvel2cartvel(self, v_rho, v_phi, rho, phi):
        vx = v_rho * math.cos(phi) - rho * v_phi * math.sin(phi)
        vy = v_rho * math.sin(phi) + rho * v_phi * math.cos(phi)
        return vx, vy

    def cartvel2polvel(self, vx, vy, x, y):
        v_rho = (x * vx + y * vy)/math.sqrt(math.pow(x, 2) + math.pow(y, 2))
        v_phi = (x * vy - vx * y)/(math.pow(x, 2) + math.pow(y, 2))
        return v_rho, v_phi

    def coord2dis(self, coord):
        dis = math.sqrt(math.pow(coord[0], 2) + math.pow(coord[1], 2))
        return dis
