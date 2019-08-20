import reeds_shepp
import math
from pdb import set_trace


class TruckPlanner():
    def __init__(self, step_size=2, rho=100):
        self.step_size = step_size
        self.rho = rho  # turning radius

    def get_path(self, present_pos, target):
        # Generates a Reeds-Shepp path
        target.append(math.pi/2)
        path = reeds_shepp.path_sample(present_pos, target, self.rho, self.step_size)
        return path


class ExcavatorPlanner():
    def __init__(self, step_size=2, rho=1):
        self.step_size = step_size
        self.rho = rho  # turning radius

    def get_path(self, present_pos, target):
        # Generates a Reeds-Shepp path
        target.append(math.pi/2)
        path = reeds_shepp.path_sample(present_pos, target, self.rho, self.step_size)
        return path
