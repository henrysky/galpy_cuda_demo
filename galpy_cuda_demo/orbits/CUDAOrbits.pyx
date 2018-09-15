import numpy as np
cimport numpy as np


cdef extern from "./CUDAOrbits.h":
    int integrate_euler_cuda(float *x, float *y, float *vx, float *vy, float *x_out, float *y_out, float *vx_out,
                             float *vy_out, int n, int steps, float dt) nogil


class CUDAOrbits:
    def __init__(self, x, y, vx, vy):
        """
        Create an array of Orbit

        :param x: x-locations
        :type x: np.ndarray
        :param y: y-locations
        :type y: np.ndarray
        :param vx: x-velocity
        :type vx: np.ndarray
        :param vy: y-velocity
        :type cy: np.ndarray
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

        self.mode = 'CUDA'

    def integrate(self, steps=1000, dt=0.1):
        """
        Orbit Integration

        :param steps: time steps to integrate
        :type steps: int
        :param dt: delta t between steps
        :type dt: float
        """

        # declare type
        cdef np.ndarray[np.float32_t, ndim=1] x_c = self.x
        cdef np.ndarray[np.float32_t, ndim=1] y_c = self.y
        cdef np.ndarray[np.float32_t, ndim=1] vx_c = self.vx
        cdef np.ndarray[np.float32_t, ndim=1] vy_c = self.vy
        cdef np.ndarray[np.float32_t, ndim=1] x_c_out = np.empty_like(np.repeat(x_c, steps))
        cdef np.ndarray[np.float32_t, ndim=1] y_c_out = np.empty_like(np.repeat(x_c, steps))
        cdef np.ndarray[np.float32_t, ndim=1] vx_c_out = np.empty_like(np.repeat(x_c, steps))
        cdef np.ndarray[np.float32_t, ndim=1] vy_c_out = np.empty_like(np.repeat(x_c, steps))

        # integrate on CUDA GPU
        integrate_euler_cuda(&x_c[0], &y_c[0], &vx_c[0], &vy_c[0], &x_c_out[0], &y_c_out[0], &vx_c_out[0], &vy_c_out[0],
                             self.num_of_obj, steps, dt)

        # CUDA should send back the whole array back to corresponding CPU memory address, we just need to put it in
        # the right python variable
        self.x = x_c_out.reshape(self.num_of_obj, steps)
        self.y = y_c_out.reshape(self.num_of_obj, steps)
        self.vx = vx_c_out.reshape(self.num_of_obj, steps)
        self.vy = vy_c_out.reshape(self.num_of_obj, steps)

        return None