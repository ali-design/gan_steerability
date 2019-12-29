from scipy.stats import truncnorm
from . import constants
import numpy as np
from utils import image

# some functions here adapted from: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb

def truncated_z_sample(batch_size, truncation=1., seed=None,
        dim_z=constants.DIM_Z):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
    return truncation * values

def one_hot(index, vocab_size=constants.VOCAB_SIZE):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output

def one_hot_if_needed(label, vocab_size=constants.VOCAB_SIZE):
    label = np.asarray(label)
    if len(label.shape) <= 1:
        label = one_hot(label, vocab_size)
    assert len(label.shape) == 2
    return label

def graph_input(graph, num_samples, seed=0, category=None, trunc=1.0):
    ''' creates z, y, trunc inputs for graph '''

    zs = truncated_z_sample(num_samples, trunc, seed, graph.dim_z)
    if category is not None:
        ys = np.array([category] * zs.shape[0])
    else:
        rnd = np.random.RandomState(seed)
        ys = rnd.randint(0,graph.vocab_size,size=zs.shape[0])

    ys = one_hot_if_needed(ys, graph.vocab_size)
    return {graph.z: zs, graph.y: ys, graph.truncation: trunc}

def imshow_unscaled_G(sess, target, return_im=False):
    np_target = sess.run(target)
    np_target_scaled = np.clip(((np_target + 1) / 2.0) * 256, 0, 255)
    im = np.concatenate(np_target_scaled, axis=0)
    image.imshow(np.uint8(im))
    if return_im:
        return im

def imshow_unscaled(target, return_im=False):
    np_target = target
    np_target_scaled = np.clip(((np_target + 1) / 2.0) * 256, 0, 255)
    im = np.concatenate(np_target_scaled, axis=0)
    image.imshow(np.uint8(im))
    if return_im:
        return im
