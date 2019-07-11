import logging
import utils.logging
from utils import util
import sys
import graphs
import time
import importlib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from IPython import embed

def train(g, graph_inputs, output_dir, save_freq=100):
    # configure logging file
    logging_file = os.path.join(output_dir, 'log.txt')
    utils.logging.configure(logging_file, append=False)

    Loss_sum = 0;
    n_epoch = 1
    optim_iter = 0
    batch_size = constants.BATCH_SIZE
    num_samples = graph_inputs[g.z].shape[0]
    loss_values = []

    # train loop
    for epoch in range(n_epoch):
        for batch_start in range(0, num_samples, batch_size):
            start_time = time.time()

            s = slice(batch_start, min(num_samples, batch_start + batch_size))
            graph_inputs_batch = util.batch_input(graph_inputs, s)
            zs_batch = graph_inputs_batch[g.z]

            # loop through the graph and target alphas returned,
            # we can have g.get_train_alpha return lists for
            # alpha_for_graph and alpha_for_target, which corresponds to
            # training with different alphas for the same zs
            # make sure that the alphas are ordered correctly though,
            # particularly for NN
            alpha_for_graph, alpha_for_target = g.get_train_alpha(zs_batch)
            if not isinstance(alpha_for_graph, list):
                alpha_for_graph = [alpha_for_graph]
                alpha_for_target = [alpha_for_target]
            for ag, at in zip(alpha_for_graph, alpha_for_target):
                feed_dict_out = graph_inputs_batch
                out_zs = g.sess.run(g.outputs_orig, feed_dict_out)

                target_fn, mask_out = g.get_target_np(out_zs, at)
                feed_dict = graph_inputs_batch
                feed_dict[g.alpha] = ag
                feed_dict[g.target] = target_fn
                feed_dict[g.mask] = mask_out
                curr_loss, _ = g.sess.run([g.loss, g.train_step], feed_dict=feed_dict)
                Loss_sum = Loss_sum + curr_loss
                loss_values.append(curr_loss)

            elapsed_time = time.time() - start_time

            logging.info('T, epc, bst, lss, a: {}, {}, {}, {}, {}'.format(
                elapsed_time, epoch, batch_start, curr_loss, at))

            if (optim_iter % save_freq == 0) and (optim_iter > 0):
                g.saver.save(g.sess, './{}/model_{}.ckpt'.format(
                    output_dir, optim_iter*batch_size),
                    write_meta_graph=False, write_state=False)

            optim_iter = optim_iter + 1

    if optim_iter > 0:
        print('average loss with this metric: ', Loss_sum/(optim_iter*batch_size))
    g.saver.save(g.sess, "./{}/model_{}_final.ckpt".format(
        output_dir, num_samples),
        write_meta_graph=False, write_state=False)
    return loss_values

if __name__ == '__main__':
    opt = TrainOptions().parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    output_dir = opt.output_dir

    graph_kwargs = util.set_graph_kwargs(opt)

    graph_util = importlib.import_module('graphs.' + opt.model + '.graph_util')
    constants = importlib.import_module('graphs.' + opt.model + '.constants')

    model = graphs.find_model_using_name(opt.model, opt.transform)
    g = model(**graph_kwargs)
    g.initialize_graph()

    # create training samples
    num_samples = opt.num_samples
    if opt.model == 'biggan' and opt.biggan.category is not None:
        graph_inputs = graph_util.graph_input(g, num_samples, seed=0, category=opt.biggan.category)
    else:
        graph_inputs = graph_util.graph_input(g, num_samples, seed=0)

    # train loop
    loss_values = train(g, graph_inputs, output_dir, opt.model_save_freq)
    loss_values = np.array(loss_values)
    np.save('./{}/loss_values.npy'.format(output_dir), loss_values)
    f, ax  = plt.subplots(figsize=(10, 4))
    ax.plot(loss_values)
    f.savefig('./{}/loss_values.png'.format(output_dir))
