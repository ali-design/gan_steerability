import os
import numpy as np
from options.vis_options import VisOptions
from utils import util
import importlib
import graphs
from utils import html

if __name__ == '__main__':
    v = VisOptions()
    v.parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples per category')
    v.parser.add_argument('--num_panels', type=int, default=7,
                        help='number of panels to show')
    v.parser.add_argument('--max_alpha', type=float,
                        help='maximum alpha value')
    v.parser.add_argument('--min_alpha', type=float,
                        help='minimum alpha value')

    opt, conf = v.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # output directory
    if opt.output_dir:
        output_dir = opt.output_dir
    else:
        output_dir = os.path.join(conf.output_dir, 'images')

    os.makedirs(output_dir, exist_ok=True)

    graph_kwargs = util.set_graph_kwargs(conf)

    graph_util = importlib.import_module('graphs.' + conf.model + '.graph_util')
    constants = importlib.import_module('graphs.' + conf.model + '.constants')

    model = graphs.find_model_using_name(conf.model, conf.transform)

    g = model(**graph_kwargs)
    g.initialize_graph()

    # restore weights
    g.saver.restore(g.sess, opt.weight_path)

    num_samples = opt.num_samples
    noise_seed = opt.noise_seed
    batch_size = constants.BATCH_SIZE

    if conf.model == 'biggan':
        graph_inputs = graph_util.graph_input(g, num_samples, seed=noise_seed,
                                              category=opt.category,
                                              trunc=opt.truncation)
    else:
        graph_inputs = graph_util.graph_input(g, num_samples, seed=noise_seed)

    for batch_start in range(0, num_samples, batch_size):
        s = slice(batch_start, min(num_samples, batch_start + batch_size))
        graph_inputs_batch = util.batch_input(graph_inputs, s)

        if 'final' in opt.weight_path:
            epochs = opt.weight_path.split('_')[-2]
        else:
            epochs = opt.weight_path.split('_')[-1].split('.')[0]

        if conf.model == 'biggan':
            filename = os.path.join(output_dir, '{}_w{}_trunc{}_seed{}'.format(
                opt.category, epochs, opt.truncation, noise_seed))
        else:
            filename = os.path.join(output_dir, 'w{}_seed{}'.format(
                epochs, noise_seed))

        if opt.max_alpha is not None and opt.min_alpha is not None:
            filename += '_max{}_min{}'.format(opt.max_alpha, opt.min_alpha)

        g.vis_image_batch(graph_inputs_batch, filename, s.start,
                          num_panels=opt.num_panels, max_alpha=opt.max_alpha,
                          min_alpha=opt.min_alpha)

    # add simple html page
    html.make_html(output_dir)



