from . import constants
import numpy as np


def z_sample(batch_size, seed=0, dim_z=constants.DIM_Z):
    rnd = np.random.RandomState(seed)
    zs = rnd.randn(batch_size, dim_z)
    return zs

def graph_input(graph, num_samples, seed=0, category=None, trunc=1.0):
    ''' creates z inputs for graph '''
    zs = z_sample(num_samples, seed, graph.dim_z)
    return {graph.z: zs}

