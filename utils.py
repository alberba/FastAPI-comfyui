import random
import time

def define_seed(seed):
    """
    Define the seed for random number generation.
    If the seed is -1, a new random seed is generated.
    Otherwise, the provided seed is used.
    """
    return random.randint(0, 2**32 - 1) if seed == -1 else seed

def get_actual_time():
    """
    Get the current time in seconds since the epoch.
    """
    return time.time()