import numpy as np
from scipy.stats import truncnorm, gaussian_kde, entropy


def compute_pdf_histogram(datapoints, ax, max_val=None, **kwargs):
    assert(type(datapoints) is np.ndarray), 'provide input as np.ndarray'
    yvals = np.ravel(datapoints)
    if max_val:
        yvals = yvals / max_val
    else:
        yvals = yvals / np.max(yvals)
    # adjust so heights (instead of area) sums to one
    weights = np.ones_like(yvals) / float(len(yvals))
    n, bins, p = ax.hist(yvals, bins=np.linspace(0, 1, 51),
                         histtype='step', weights=weights, **kwargs)
    return n, bins, p

def compute_pdf_kde(datapoints, max_x=None, normalize=False):
    assert(type(datapoints) is np.ndarray), 'provide input as np.ndarray'
    yvals = np.ravel(datapoints)
    # construct xvals
    if max_x:
        xvals = np.arange(max_x)
    else:
        xvals = np.arange(min(yvals), max(yvals))

    # normalize?
    if normalize:
        yvals /= np.max(xvals)
        # xvals /= np.max(xvals)
        xvals = np.linspace(0, 1, 101)

    # get kde estimate
    yvals = gaussian_kde(yvals).evaluate(xvals)
    return xvals, yvals

def area_of_intersection(data_yval, yval):
    return sum([min(ii, jj) for (ii, jj) in zip(data_yval, yval)])

def compute_entropy(datapoints, max_val):
    assert(type(datapoints) is np.ndarray), 'provide input as np.ndarray'
    yvals = np.ravel(datapoints)
    yvals = yvals / max_val
    # adjust so heights (instead of area) sums to one
    weights = np.ones_like(yvals) / float(len(yvals))
    n, bins = np.histogram(yvals, bins=np.linspace(0, 1, 51), weights=weights)
    return entropy(n)

def pick_color(path):
    colors = [[40, 71, 145], [59, 131, 182], [97, 186, 210], [172, 207, 122]]
    colors = [np.array(c) / 255 for c in colors]
    if 'color' in path:
        return colors[0]
    elif 'shiftx' in path:
        return colors[1]
    elif 'shifty' in path:
        return colors[2]
    elif 'zoom' in path:
        return colors[3]
    return colors[0]

