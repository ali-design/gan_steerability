import os
import numpy as np
from options.vis_options import VisOptions
from utils import util
import importlib
import graphs
import seaborn as sns
import matplotlib.pyplot as plt

def draw_distribution(g, output_dir, xlabel, num_samples, channel):

    if g.Nsliders == 1:
        print("Using default channel value for this transform")
        print("Channel value used for color transforms")
        channel = None

    output_path = os.path.join(output_dir, 'channel_{}'.format(channel))
    os.makedirs(output_path, exist_ok=True)

    # get transformation attributes
    if os.path.isfile(os.path.join(output_path, 'model_samples.npy')):
        print("Loading model distributions")
        model_distribution = np.load(os.path.join(
            output_path, 'model_samples.npy'))
    else:
        model_distribution = g.get_distribution(num_samples, channel=channel)
        np.save(os.path.join(output_path, 'model_samples.npy'),
                model_distribution)

    alphas = g.test_alphas()

    sns.set(color_codes=True)
    sns.set(font_scale=3)
    sns.set_style("whitegrid")
    sns.set_palette("hls", len(model_distribution))
    midpt = len(model_distribution) // 2

    f, ax = plt.subplots(figsize=(15, 12))
    for i, (alpha, samples) in enumerate(zip(alphas, model_distribution)):
        if i == midpt:
            continue
        sns.kdeplot(samples, ax=ax, linewidth=3.0,
                    label='alpha={}'.format(alpha))
    # plot the untransformed distribution 
    sns.kdeplot(model_distribution[midpt], ax=ax, linewidth=5.0,
                color='darkblue', label='model')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('PDF')
    f.savefig(os.path.join(output_path, 'distribution.png'), bbox_inches='tight')

if __name__ == '__main__':
    v = VisOptions()
    v.parser.add_argument('--num_samples', type=int, default=1000,
                        help='number of samples for distribution')
    v.parser.add_argument('--channel', type=None,
                          help='which color channel, for color transformation, using channel=None computes luminance')

    opt, conf = v.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # output directory
    if opt.output_dir:
        output_dir = opt.output_dir
    else:
        output_dir = os.path.join(conf.output_dir, 'distribution')

    os.makedirs(output_dir, exist_ok=True)

    # create graph
    graph_kwargs = util.set_graph_kwargs(conf)
    graph_util = importlib.import_module('graphs.' + conf.model + '.graph_util')
    constants = importlib.import_module('graphs.' + conf.model + '.constants')

    model = graphs.find_model_using_name(conf.model, conf.transform)

    g = model(**graph_kwargs)
    g.initialize_graph()

    # restore weights
    g.saver.restore(g.sess, opt.weight_path)

    # valid for these 4 transformations
    assert(conf.transform in ['zoom', 'shiftx', 'shifty', 'color'])
    xlabel=dict(zoom='Area', shiftx='Center X', shifty='Center Y',
                color='Pixel Intensity')

    draw_distribution(g, output_dir, xlabel[conf.transform],
                      opt.num_samples, opt.channel)
