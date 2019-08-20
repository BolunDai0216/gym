import math
import numpy as np
import gym

MAX_ANGULAR_VEL = math.tan(math.pi/6)


class OptimalDumpingPath():
    def __init__(self):
        L = 20
        max_turning_degree = math.pi/6
        min_rad = L / math.tan(max_turning_degree)
        env = gym.make('Truck-v4')

        self.obsv_list = np.ones((0, 5))

        for i in range(1):
            env.reset()
            done = False
            flag = True
            while not done:
                if flag:
                    if env.car.position[0] <= -min_rad:
                        obsv, reward, done, info = env.step([0, 1])
                    if env.car.position[0] > -min_rad and env.car.position[0] < 0:
                        if env.car.angle > (3*math.pi/2+20/min_rad):
                            obsv, reward, done, info = env.step([-1, 1])
                        else:
                            obsv, reward, done, info = env.step([-1, -1])
                    if env.car.position[0] >= 0:
                        flag = False
                        obsv, reward, done, info = env.step([0, -1])
                else:
                    obsv, reward, done, info = env.step([0, -1])
                self.obsv_list = np.vstack((self.obsv_list, obsv))

    def get_init_state(self):
        shape = self.obsv_list.shape
        pick = np.random.choice(range(shape[0]), 1)[0]
        init_state = self.obsv_list[pick, :]
        init_state = np.array([init_state[0]*200, init_state[1]*200,
                               init_state[2]*(2*math.pi), init_state[3]*20,
                               init_state[4]*MAX_ANGULAR_VEL])

        return init_state


def main():
    path = OptimalDumpingPath()
    path.get_init_state()


if __name__ == "__main__":
    main()
