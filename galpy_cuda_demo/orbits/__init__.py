from .CUDAOrbits import CUDAOrbits
from .npOrbits import npOrbits


def Orbits(x, y, vx, vy, mode='cuda'):
    """
    Create Orbits array orbit, return Orbits instance depending on the mode

    :param x: x-locations in AU
    :type x: np.ndarray
    :param y: y-locations in AU
    :type y: np.ndarray
    :param vx: x-velocity in AU/yr
    :type vx: np.ndarray
    :param vy: y-velocity in AU/yr
    :type vy: np.ndarray
    :param mode: 'cuda' to use CUDA GPU or 'cpu' to use numpy CPU
    :type mode: str
    """
    if mode.lower() == 'cuda':
        return CUDAOrbits(x, y, vx, vy)
    elif mode.lower() == 'cpu':
        return npOrbits(x, y, vx, vy)
    else:
        raise ValueError("Mode can only be either 'CUDA' or 'CPU'")
