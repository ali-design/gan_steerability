from . import constants
import numpy as np


def z_sample(batch_size, seed=0, dim_z=constants.DIM_Z):
    rnd = np.random.RandomState(seed)
    zs = rnd.randn(batch_size, dim_z)
    return zs

def graph_input(graph, num_samples, seed=0, **kwargs):
    ''' creates z inputs for graph '''
    zs = z_sample(num_samples, seed, graph.dim_z)
    labels = np.zeros((num_samples, 0))
    return {graph.z: zs, graph.labels: labels}

