import math
import numpy as np
from pdb import set_trace


class DemoTrack():
    def __init__(self):
        self.truck_trjctry = np.zeros((1000, 3))
        self.excav_trjctry = np.zeros((1000, 2))
        self.get_track()

        _truck_xs = self.truck_trjctry[:, 0].tolist()
        _truck_ys = self.truck_trjctry[:, 1].tolist()
        self.truck_xys = list(zip(_truck_xs, _truck_ys))

        _excvtr_xs = self.excav_trjctry[:, 0].tolist()
        _excvtr_ys = self.excav_trjctry[:, 1].tolist()
        self.excvtr_xys = list(zip(_excvtr_xs, _excvtr_ys))

    def get_track(self):
        for i in range(1000):
            if i <= 375:
                truck_angle = math.pi * i * 0.5/375
                self.truck_trjctry[i, :] = [
                    375 - 325 * math.cos(truck_angle), 700 - 325 * math.sin(truck_angle), truck_angle]
            elif i <= 500:
                truck_angle = math.pi * (i - 375) * 0.5/125
                self.truck_trjctry[i, :] = [
                    375 - 75 * math.sin(truck_angle), 300 + 75 * math.cos(truck_angle), truck_angle + math.pi * 0.5]
            elif i <= 550:
                self.truck_trjctry[i, :] = self.truck_trjctry[500, :]
            else:
                self.truck_trjctry[i, :] = [300, 300 + (i - 550), 180]

            if i <= 500:
                self.excav_trjctry[i, :] = [
                    300 - 130 * math.cos(math.pi * i / 1000), 100 + 130 * math.sin(math.pi * i / 1000)]
            else:
                self.excav_trjctry[i, :] = [300, 230]


def main():
    demo = DemoTrack()
    print(demo.truck_trjctry)
    set_trace()


if __name__ == "__main__":
    main()
