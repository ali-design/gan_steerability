import io
import numpy as np
import PIL.Image
import IPython.display
from scipy.stats import truncnorm, gaussian_kde, entropy
import cv2
import os

# adapted from: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb
def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows*H, cols*W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid

# adapted from: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb
def imshow(a, format='png', jpeg_fallback=True, filename=None):
    ''' shows image in jupyter notebook '''
    a = np.asarray(a, dtype=np.uint8)
    str_file = io.BytesIO()
    PIL.Image.fromarray(a).save(str_file, format)
    im_data = str_file.getvalue()
    try:
        disp = IPython.display.display(IPython.display.Image(im_data))
        if filename:
            size = (a.shape[1]//2, a.shape[0]//2)
            im = PIL.Image.fromarray(a)
            im.thumbnail(size,PIL.Image.ANTIALIAS)
            im.save('{}.{}'.format(filename, format))
    except IOError:
        if jpeg_fallback and format != 'jpeg':
            print ('Warning: image was too large to display in '
                   'format "{}"; trying jpeg instead.'.format(format))
            return imshow(a, format='jpeg')
        else:
            raise
    return disp

def save_im(a, filename, format='png'):
    a = np.asarray(a, dtype=np.uint8)
    size = (a.shape[1]//2, a.shape[0]//2)
    im = PIL.Image.fromarray(a)
    im.thumbnail(size,PIL.Image.ANTIALIAS)
    im.save('{}.{}'.format(filename, format))


def load_and_resize_imagenet_image(img_path, size=(256, 256)):
    img = cv2.imread(img_path)
    img = img[:,:,::-1] # BGR2RGB
    [row, col, _] = img.shape
    dim = min(row, col)
    if col == row:
        pass
    elif dim == row:
        col_start = (col-dim)//2 # off-by-one rounding
        col_start = col_start if col_start == 0 else np.random.choice(col_start)
        img = img[:, col_start:col_start+dim, :]
    elif dim == col:
        row_start = (row-dim)//2 # off-by-one rounding
        row_start = row_start if row_start == 0 else np.random.choice(row_start)
        img = img[row_start:row_start+dim, :, :]
    [row, col, _] = img.shape
    assert(row == col)
    img = cv2.resize(img, size)
    return img


