import numpy as np


class npOrbits:
    def __init__(self, x, y, vx, vy):
        """
        Create an array of Orbit

        :param x: x-locations in AU
        :type x: np.ndarray
        :param y: y-locations in AU
        :type y: np.ndarray
        :param vx: x-velocity in AU/yr
        :type vx: np.ndarray
        :param vy: y-velocity in AU/yr
        :type vy: np.ndarray
        """
        # make sure CUDA received float32 ndarray
        self.x = np.array(x).astype(np.float32)
        self.y = np.array(y).astype(np.float32)
        self.vx = np.array(vx).astype(np.float32)
        self.vy = np.array(vy).astype(np.float32)

        # number of object in this orbits
        self.num_of_obj = self.x.shape[0]

        # assert to make sure, CUDA is lame and crash system otherwise
        assert self.x.shape[0] == self.y.shape[0] == self.vx.shape[0] == self.vy.shape[0]
        assert len(self.x.shape) == len(self.y.shape) == len(self.vx.shape) == len(self.vy.shape)

        self.mode = 'Numpy'

    def integrate(self, steps=1000, dt=0.1):
        """
        Orbit Integration

        :param steps: time steps to integrate
        :type steps: int
        :param dt: delta t between steps
        :type dt: float
        """
        M_s = np.float32(1.)  # solar mass
        G = np.float32(39.5)  # newtonian constant of gravitation

        R3 = lambda x, y: (x ** 2 + y ** 2) ** (3 / 2)

        x_result = np.empty((self.num_of_obj, steps)).astype(np.float32)
        y_result = np.empty((self.num_of_obj, steps)).astype(np.float32)
        vx_result = np.empty((self.num_of_obj, steps)).astype(np.float32)
        vy_result = np.empty((self.num_of_obj, steps)).astype(np.float32)

        x_result[:, 0] = self.x
        y_result[:, 0] = self.y
        vx_result[:, 0] = self.vx
        vy_result[:, 0] = self.vy

        for t in range(0, steps - 1):
            vx_result[:, t + 1] = vx_result[:, t] - dt * (G * M_s * x_result[:, t] / R3(x_result[:, t], y_result[:, t]))
            vy_result[:, t + 1] = vy_result[:, t] - dt * (G * M_s * y_result[:, t] / R3(x_result[:, t], y_result[:, t]))
            x_result[:, t + 1] = x_result[:, t] + dt * vx_result[:, t + 1]
            y_result[:, t + 1] = y_result[:, t] + dt * vy_result[:, t + 1]

        self.x = x_result
        self.y = y_result
        self.vx = vx_result
        self.vy = vy_result

        return None

    @property
    def R(self):
        return np.sqrt(self.x**2 + self.y**2)