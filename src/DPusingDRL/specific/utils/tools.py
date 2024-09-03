import numpy as np

def ssa(angle):
    """
    This function converts any angle into the range [-π, π].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
