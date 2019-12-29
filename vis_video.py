import os
import numpy as np
from options.vis_options import VisOptions
from utils import util, video
import importlib
import graphs
from utils import html

if __name__ == '__main__':
    v = VisOptions()
    v.parser.add_argument('--sample', type=int, default=0,
                        help='which random sample to take ' +
                          'e.g. the 10th sample from seed 0')
    v.parser.add_argument('--num_frames', type=int, default=100,
                        help='number of frames')
    v.parser.add_argument('--max_alpha', type=float,
                        help='maximum alpha value')
    v.parser.add_argument('--min_alpha', type=float,
                        help='minimum alpha value')
    v.parser.add_argument('--channel', type=None,
                          help='which color channel, for color transformation')
    v.parser.add_argument('--filename', default='',
                          help='output filename, otherwise uses default naming')

    opt, conf = v.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # output directory
    if opt.output_dir:
        output_dir = opt.output_dir
    else:
        output_dir = os.path.join(conf.output_dir, 'videos')

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

    sample = opt.sample
    noise_seed = opt.noise_seed

    if conf.model == 'biggan':
        graph_inputs = graph_util.graph_input(g, sample+1, seed=noise_seed,
                                              category=opt.category,
                                              trunc=opt.truncation)
        filename = os.path.join(output_dir, '{}_seed{}_sample{}.mp4'
                                .format(opt.category, noise_seed, sample))
    else:
        graph_inputs = graph_util.graph_input(g, sample+1, seed=noise_seed)
        filename = os.path.join(output_dir, 'seed{}_sample{}.mp4'
                                .format(noise_seed, sample))
    if opt.filename:
        # override default filename
        filename = opt.filename

    # get the appropriate sample from the inputs
    for k,v in graph_inputs.items():
        if isinstance(v, np.ndarray) and v.shape[0] == sample+1:
            graph_inputs[k] = v[sample][None]

    # create video
    video.make_video(g, graph_inputs, filename, num_frames=opt.num_frames,
                     max_alpha=opt.max_alpha, min_alpha=opt.min_alpha,
                     channel=opt.channel)
