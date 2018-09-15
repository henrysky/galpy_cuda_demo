from .CUDAOrbits import CUDAOrbits
from .npOrbits import npOrbits


def Orbits(x, y, vx, vy, mode='cuda'):
    """
    Create Orbits array orbit, return Orbits instance depending on the mode

    :param x: x-locations
    :type x: np.ndarray
    :param y: y-locations
    :type y: np.ndarray
    :param vx: x-velocity
    :type vx: np.ndarray
    :param vy: y-velocity
    :type cy: np.ndarray
    """
    if mode.lower() == 'cuda':
        return CUDAOrbits(x, y, vx, vy)
    elif mode.lower() == 'cpu':
        return npOrbits(x, y, vx, vy)
    else:
        raise ValueError("Mode can only be either 'CUDA' or 'CPU'")
