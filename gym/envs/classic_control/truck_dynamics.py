import math
import numpy as np

WHEEL_DIS = 20  # m
TRUCK_LENGTH = 24  # m
TRUCK_WIDTH = 15  # m

MAX_STEERING_ANGLE = 30  # deg
MIN_STEERING_ANGLE = -MAX_STEERING_ANGLE  # deg

MAX_SPEED = 20  # m/sec
MAX_ACCELERATION = 4  # m/sec^2
TIME_INTERVAL = 0.1  # sec


class Truck():
    def __init__(self, init_x, init_y, init_angle, xlim=[-200, 200], ylim=[-200, 200]):
        self.angle = init_angle  # in radian
        self.position = [init_x, init_y]  # rear position
        self.velocity = 0
        self.acceleration = 0
        self.angular_velocity = 0
        self.dt = TIME_INTERVAL
        self.xlim = xlim
        self.ylim = ylim

    def step(self, steer, gas):
        """
        Following the transisiton equation:
                v_x = speed * cos(angle)
                v_y = speed * sin(angle)
            v_angle = (speed/wheel_dis) * tan(steering_angle)

        source: http://planning.cs.uiuc.edu/node658.html

        Also the following kinematics equation:
            acceleration = acceleration
                   speed = init_speed + acceleration * time_interval
        """
        FLAG = False
        steer = np.clip(steer, -1, +1)
        gas = np.clip(gas, -1, +1)
        self.acceleration = gas * MAX_ACCELERATION
        self.velocity += self.acceleration * self.dt

        # Make sure it is within bound
        if self.velocity >= MAX_SPEED:
            self.velocity = MAX_SPEED
        if self.velocity <= -MAX_SPEED/3:
            self.velocity = -MAX_SPEED/3

        # Update position
        self.position[0] += self.dt * self.velocity * math.cos(self.angle)
        self.position[1] += self.dt * self.velocity * math.sin(self.angle)

        if self.position[0] > self.xlim[1]:
            self.position[0] = self.xlim[1]
            FLAG = True
        elif self.position[0] < self.xlim[0]:
            self.position[0] = self.xlim[0]
            FLAG = True

        if self.position[1] > self.ylim[1]:
            self.position[1] = self.ylim[1]
            FLAG = True
        elif self.position[1] < self.ylim[0]:
            self.position[1] = self.ylim[0]
            FLAG = True

        turning_angle = steer * MAX_STEERING_ANGLE
        turning_angle_rad = math.pi * turning_angle / 180

        # Make sure there won't be any numerical issues caused by inf
        self.angular_velocity = 0
        if math.fabs(turning_angle) >= 1:
            self.angular_velocity = self.velocity*math.tan(turning_angle_rad)/WHEEL_DIS

        # Update angle
        self.angle += self.dt * self.angular_velocity

        return self.position, self.angle, self.velocity, self.angular_velocity, FLAG
